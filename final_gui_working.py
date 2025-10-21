#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optiver-style Fruit Trading Simulator (GUI Version) - Simplified Working Version

A graphical trading game with live countdown timers, separate buy/sell areas,
and clickable buttons for trading.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import time
import threading
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, scrolledtext

# Use existing models.py
from models import GameConfig, Tally, SideBet, SideBetFn, Position


# =============================
# Math helpers
# =============================
def _poisson(lam: float) -> int:
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1


def _noisy_int_quote(ev: float, cfg: GameConfig) -> int:
    if ev < 0:
        ev = 0.0
    if random.random() < cfg.shock_prob:
        eps = random.gauss(0.0, cfg.sigma_shock)
    else:
        eps = random.gauss(0.0, cfg.sigma_normal)
    price = ev * (1.0 + eps)
    return max(0, int(round(price)))


def main_ev_product(current: Tally, mu_oranges: float, mu_lemons: float) -> float:
    o = current.oranges_total
    lemons = current.lemons_total
    return o * lemons + o * mu_lemons + lemons * mu_oranges + mu_oranges * mu_lemons


def ev_2_pow_oranges_A(current: Tally, mu_future_A_oranges: float) -> float:
    return (2 ** current.oranges_A) * math.exp(mu_future_A_oranges)


def ev_sum_oranges_lemons(current: Tally, mu_oranges: float, mu_lemons: float) -> float:
    return current.oranges_total + current.lemons_total + mu_oranges + mu_lemons


def ev_exp_oranges(current: Tally, mu_oranges: float, mu_lemons: float) -> float:
    """Expected value of 2^(total oranges)"""
    expected_oranges = current.oranges_total + mu_oranges
    return 2 ** expected_oranges


def ev_exp_lemons(current: Tally, mu_oranges: float, mu_lemons: float) -> float:
    """Expected value of 2^(total lemons)"""
    expected_lemons = current.lemons_total + mu_lemons
    return 2 ** expected_lemons


def ev_exp_sum(current: Tally, mu_oranges: float, mu_lemons: float) -> float:
    """Expected value of 2^(oranges + lemons)"""
    expected_sum = current.oranges_total + current.lemons_total + mu_oranges + mu_lemons
    return 2 ** expected_sum


def ev_oranges_A_linear(current: Tally, mu_future_A_oranges: float) -> float:
    return current.oranges_A + mu_future_A_oranges


# =============================
# Engine
# =============================
class FruitEngine:
    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        if cfg.seed is not None:
            random.seed(cfg.seed)
        self.reset()

    def reset(self):
        self.tick = 0
        self.tally = Tally()
        self.history: List[Tally] = []
        self.pos = Position()
        self.sidebets: Dict[str, SideBet] = {}
        self.archive_sidebets: Dict[str, SideBet] = {}
        self.next_sb_id = 1
        self.remaining_mu_oranges = self.cfg.lambda_oranges * self.cfg.ticks
        self.remaining_mu_lemons = self.cfg.lambda_lemons * self.cfg.ticks
        self.game_start_time = None
        self.next_tick_time = None
        self.last_sidebet_spawn_time = None

    def simulate_tick(self):
        if self.tick >= self.cfg.ticks:
            return
        inc_oranges = _poisson(self.cfg.lambda_oranges)
        inc_lemons = _poisson(self.cfg.lambda_lemons)

        a_oranges = sum(1 for _ in range(inc_oranges) 
                       if random.random() < self.cfg.team_A_strength_orange)
        b_oranges = inc_oranges - a_oranges
        a_lemons = sum(1 for _ in range(inc_lemons) 
                      if random.random() < self.cfg.team_A_strength_lemon)
        b_lemons = inc_lemons - a_lemons

        self.tally.oranges_A += a_oranges
        self.tally.oranges_B += b_oranges
        self.tally.lemons_A += a_lemons
        self.tally.lemons_B += b_lemons

        self.remaining_mu_oranges = max(0.0, self.remaining_mu_oranges - 
                                       self.cfg.lambda_oranges)
        self.remaining_mu_lemons = max(0.0, self.remaining_mu_lemons - 
                                      self.cfg.lambda_lemons)

        self.tick += 1
        self.history.append(self.tally.copy())

    def current_quotes(self) -> Tuple[int, Dict[str, int]]:
        """Calculate current market quotes."""
        t = self.tally
        mu_o = self.remaining_mu_oranges
        mu_l = self.remaining_mu_lemons

        # Main market price (less frequent updates)
        main_ev = main_ev_product(t, mu_o, mu_l)
        main_px = _noisy_int_quote(main_ev, self.cfg)

        # Side bet prices
        side_px = {}
        for sid, sb in self.sidebets.items():
            if sb.price == 0:  # Calculate price if not set
                try:
                    if sb.kind == 'sum':
                        ev = ev_sum_oranges_lemons(t, mu_o, mu_l)
                    elif sb.kind == 'exp_oranges':
                        ev = ev_exp_oranges(t, mu_o, mu_l)
                    elif sb.kind == 'exp_lemons':
                        ev = ev_exp_lemons(t, mu_o, mu_l)
                    elif sb.kind == 'exp_sum':
                        ev = ev_exp_sum(t, mu_o, mu_l)
                    elif sb.kind == 'exp_team_a':
                        ev = 2 ** min(t.oranges_A + t.lemons_A + 
                                     (mu_o + mu_l) * self.cfg.team_A_strength_orange, 20)
                    elif sb.kind == 'exp_team_b':
                        ev = 2 ** min(t.oranges_B + t.lemons_B + 
                                     (mu_o + mu_l) * (1 - self.cfg.team_A_strength_orange), 20)
                    else:
                        ev = sb.payoff(t)  # Use current payoff as estimate
                    
                    sb.price = _noisy_int_quote(ev, self.cfg)
                except Exception as e:
                    print(f"Error pricing side bet {sid}: {e}")
                    sb.price = 1
            
            side_px[sid] = sb.price

        return main_px, side_px

    def check_expired_sidebets(self):
        """Remove expired side bets."""
        now = datetime.now()
        expired = []
        for sid, sb in self.sidebets.items():
            if now >= sb.expiry_time:
                expired.append(sid)

        for sid in expired:
            sb = self.sidebets.pop(sid)
            self.archive_sidebets[sid] = sb

    def maybe_spawn_sidebet(self):
        """Spawn a new side bet with some probability."""
        if len(self.sidebets) >= self.cfg.max_concurrent_sidebets:
            return

        now = datetime.now()
        if (self.last_sidebet_spawn_time and 
            (now - self.last_sidebet_spawn_time).total_seconds() < 3):
            return

        # Use sidebet_prob_per_spawn instead of sidebet_spawn_prob
        if random.random() > getattr(self.cfg, 'sidebet_prob_per_spawn', 0.85):
            return

        # Simplified side bet choices
        choices = [
            ('sum', 'All Fruit (🍊+🍎)', 
             lambda t: t.oranges_total + t.lemons_total),
            ('exp_oranges', '2^🍊', 
             lambda t: min(int(2 ** min(t.oranges_total, 20)), 999999)),
            ('exp_lemons', '2^🍎', 
             lambda t: min(int(2 ** min(t.lemons_total, 20)), 999999)),
            ('exp_sum', '2^(🍊+🍎)',
             lambda t: min(int(2 ** min(t.oranges_total + t.lemons_total, 20)), 999999)),
        ]
        kind, desc, fn = random.choice(choices)
        sid = f"SB{self.next_sb_id}"
        self.next_sb_id += 1

        spawn_time = now
        duration = max(5, self.cfg.sidebet_duration_seconds + 
                      random.randint(-getattr(self.cfg, 'sidebet_duration_variance', 5),
                                     getattr(self.cfg, 'sidebet_duration_variance', 5)))
        expiry_time = now + timedelta(seconds=duration)

        sb_obj = SideBet(
            sid=sid,
            kind=kind,
            description=desc,
            payoff=fn,
            price=0,
            expiry_time=expiry_time,
            spawn_time=spawn_time
        )
        self.sidebets[sid] = sb_obj
        self.last_sidebet_spawn_time = now

    def settle_value(self, tally: Optional[Tally] = None) -> int:
        t = tally or self.tally
        return int(t.oranges_total * t.lemons_total)

    def side_payoff(self, sb: SideBet, tally: Optional[Tally] = None) -> int:
        t = tally or self.tally
        return int(sb.payoff(t))

    def get_sidebet(self, sid: str) -> Optional[SideBet]:
        return self.sidebets.get(sid) or self.archive_sidebets.get(sid)


# =============================
# Past games generator
# =============================
def simulate_past_games(cfg: GameConfig, n: int) -> List[Tally]:
    out: List[Tally] = []
    for _ in range(n):
        eng = FruitEngine(cfg)
        eng.reset()
        for _ in range(cfg.ticks):
            eng.simulate_tick()
        out.append(eng.tally.copy())
    return out


# =============================
# GUI
# =============================
class TradingGUI:
    def __init__(self, root: tk.Tk, cfg: GameConfig):
        self.root = root
        self.cfg = cfg
        self.engine = FruitEngine(cfg)
        self.past = simulate_past_games(cfg, cfg.past_games)
        
        self.game_running = False
        self.update_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self.last_main_px = 0
        self.last_side_px: Dict[str, int] = {}
        self.game_paused = False
        self.last_trade_info = None
        self.status_message = None
        self.last_tally = Tally(0, 0, 0, 0)
        
        self.root.title("Market Making Trading Simulator")
        self.root.geometry("1400x900")
        self.root.configure(bg='#ffffff')
        self.root.minsize(1200, 800)
        
        self.setup_ui()
        self.show_past_games()
        
    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='#ffffff')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar for trade feedback (top)
        self.status_bar = tk.Label(main_frame, text="Ready to trade", 
                                   font=('Arial', 13, 'bold'), bg='#e8f5e8', 
                                   fg='#006600', relief=tk.SUNKEN, bd=1,
                                   anchor=tk.W, padx=10, pady=8)
        self.status_bar.pack(fill=tk.X, pady=(0, 5))
        
        # Progress bar (shows game completion)
        progress_frame = tk.Frame(main_frame, bg='#ffffff')
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(progress_frame, text="Time Remaining:", 
                font=('Arial', 12, 'bold'), bg='#ffffff', 
                fg='#000000').pack(side=tk.LEFT, padx=5)
        
        # Countdown timer in MM:SS format (game time, 6x real time)
        self.time_label = tk.Label(progress_frame, text="90:00", 
                                   font=('Arial', 16, 'bold'), bg='#ffffff', 
                                   fg='#cc0000', width=8)
        self.time_label.pack(side=tk.LEFT, padx=10)
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=400, 
                                           mode='determinate', 
                                           maximum=self.cfg.ticks)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Create wrapper for center area with side bets on left/right
        center_wrapper = tk.Frame(main_frame, bg='#ffffff')
        center_wrapper.pack(fill=tk.BOTH, expand=True)
        
        # Left side: Side bets (left panel)
        left_sidebets_frame = tk.LabelFrame(center_wrapper, text="SIDE BETS", 
                                           font=('Arial', 12, 'bold'), 
                                           bg='#ffffff', fg='#8800cc', 
                                           relief=tk.RIDGE, bd=2, 
                                           width=300, height=500)
        left_sidebets_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, 
                                padx=(0, 5))
        left_sidebets_frame.pack_propagate(False)
        
        # Scrollable canvas for left side bets
        left_canvas = tk.Canvas(left_sidebets_frame, bg='#ffffff', 
                               highlightthickness=0)
        left_scrollbar = tk.Scrollbar(left_sidebets_frame, orient="vertical", 
                                     command=left_canvas.yview)
        self.left_sidebets_container = tk.Frame(left_canvas, bg='#ffffff')
        
        self.left_sidebets_container.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )
        
        left_canvas.create_window((0, 0), window=self.left_sidebets_container, 
                                 anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Center: Main market
        main_market_frame = tk.LabelFrame(center_wrapper, text="MAIN MARKET", 
                                         font=('Arial', 14, 'bold'), 
                                         bg='#ffffff', fg='#0066cc', 
                                         relief=tk.RIDGE, bd=3,
                                         width=500)
        main_market_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, 
                              padx=5)
        main_market_frame.pack_propagate(False)
        
        # Game state display (top of main market)
        state_frame = tk.Frame(main_market_frame, bg='#ffffff')
        state_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Current fruit counts
        fruit_frame = tk.Frame(state_frame, bg='#ffffff')
        fruit_frame.pack(fill=tk.X)
        
        # Team A
        team_a_frame = tk.Frame(fruit_frame, bg='#ffffff')
        team_a_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(team_a_frame, text="Team A", 
                font=('Arial', 12, 'bold'), bg='#ffffff', 
                fg='#ff6600').pack(anchor=tk.W)
        
        self.oranges_a_label = tk.Label(team_a_frame, text="🍊 0", 
                                       font=('Arial', 16, 'bold'), 
                                       bg='#ffffff', fg='#000000')
        self.oranges_a_label.pack(anchor=tk.W, pady=2)
        
        self.lemons_a_label = tk.Label(team_a_frame, text="🍎 0", 
                                      font=('Arial', 16, 'bold'), 
                                      bg='#ffffff', fg='#000000')
        self.lemons_a_label.pack(anchor=tk.W, pady=2)
        
        # Separator
        separator = tk.Frame(fruit_frame, bg='#cccccc', width=2)
        separator.pack(side=tk.LEFT, fill=tk.Y, padx=20)
        
        # Team B
        team_b_frame = tk.Frame(fruit_frame, bg='#ffffff')
        team_b_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(team_b_frame, text="Team B", 
                font=('Arial', 12, 'bold'), bg='#ffffff', 
                fg='#0066cc').pack(anchor=tk.W)
        
        self.oranges_b_label = tk.Label(team_b_frame, text="🍊 0", 
                                       font=('Arial', 16, 'bold'), 
                                       bg='#ffffff', fg='#000000')
        self.oranges_b_label.pack(anchor=tk.W, pady=2)
        
        self.lemons_b_label = tk.Label(team_b_frame, text="🍎 0", 
                                      font=('Arial', 16, 'bold'), 
                                      bg='#ffffff', fg='#000000')
        self.lemons_b_label.pack(anchor=tk.W, pady=2)
        
        # Current market value
        value_frame = tk.Frame(main_market_frame, bg='#e6f3ff')
        value_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        tk.Label(value_frame, text="Current Payoff (🍊 × 🍎):", 
                font=('Arial', 14, 'bold'), bg='#e6f3ff', 
                fg='#0066cc').pack(side=tk.LEFT, pady=8, padx=8)
        
        self.current_value_label = tk.Label(value_frame, text="0", 
                                           font=('Arial', 18, 'bold'), 
                                           bg='#e6f3ff', fg='#cc0000')
        self.current_value_label.pack(side=tk.RIGHT, pady=8, padx=8)
        
        # Main market price
        price_frame = tk.Frame(main_market_frame, bg='#ffffcc')
        price_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        tk.Label(price_frame, text="Market Price:", 
                font=('Arial', 16, 'bold'), bg='#ffffcc', 
                fg='#cc6600').pack(side=tk.LEFT, pady=10, padx=10)
        
        self.main_price_label = tk.Label(price_frame, text="0", 
                                        font=('Arial', 20, 'bold'), 
                                        bg='#ffffcc', fg='#000000')
        self.main_price_label.pack(side=tk.RIGHT, pady=10, padx=10)
        
        # Position display
        position_frame = tk.Frame(main_market_frame, bg='#f0f0f0')
        position_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        tk.Label(position_frame, text="Your Position:", 
                font=('Arial', 14, 'bold'), bg='#f0f0f0', 
                fg='#666666').pack(side=tk.LEFT, pady=8, padx=8)
        
        self.position_label = tk.Label(position_frame, text="0", 
                                      font=('Arial', 16, 'bold'), 
                                      bg='#f0f0f0', fg='#000000')
        self.position_label.pack(side=tk.RIGHT, pady=8, padx=8)
        
        # Trading controls
        main_controls = tk.Frame(main_market_frame, bg='#ffffff')
        main_controls.pack(side=tk.BOTTOM, pady=15)
        
        tk.Label(main_controls, text="Quantity:", 
                font=('Arial', 14, 'bold'), bg='#ffffff', 
                fg='#000000').pack(side=tk.LEFT, padx=5)
        
        self.main_qty_entry = tk.Entry(main_controls, font=('Arial', 16, 'bold'), 
                                      width=8, justify='center')
        self.main_qty_entry.pack(side=tk.LEFT, padx=5)
        self.main_qty_entry.insert(0, "1")
        
        buy_btn = tk.Button(main_controls, text="BUY", font=('Arial', 16, 'bold'), 
                           bg='#00ff00', fg='black', width=10, height=2,
                           activebackground='#00cc00', relief=tk.RAISED, bd=3,
                           cursor='hand2',
                           command=lambda: self.execute_trade('buy', 'main'))
        buy_btn.pack(side=tk.LEFT, padx=8)
        
        sell_btn = tk.Button(main_controls, text="SELL", font=('Arial', 16, 'bold'), 
                            bg='#ff4400', fg='#000000', width=10, height=2,
                            activebackground='#cc3300', relief=tk.RAISED, bd=3,
                            cursor='hand2',
                            command=lambda: self.execute_trade('sell', 'main'))
        sell_btn.pack(side=tk.LEFT, padx=8)
        
        # Right side: More Side bets (right panel)
        right_sidebets_frame = tk.LabelFrame(center_wrapper, text="SIDE BETS", 
                                            font=('Arial', 12, 'bold'), 
                                            bg='#ffffff', fg='#8800cc', 
                                            relief=tk.RIDGE, bd=2,
                                            width=300, height=500)
        right_sidebets_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, 
                                 padx=(5, 0))
        right_sidebets_frame.pack_propagate(False)
        
        # Scrollable canvas for side bets
        right_canvas = tk.Canvas(right_sidebets_frame, bg='#ffffff', 
                                highlightthickness=0)
        right_scrollbar = tk.Scrollbar(right_sidebets_frame, orient="vertical", 
                                      command=right_canvas.yview)
        self.right_sidebets_container = tk.Frame(right_canvas, bg='#ffffff')
        
        self.right_sidebets_container.bind(
            "<Configure>",
            lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        )
        
        right_canvas.create_window((0, 0), window=self.right_sidebets_container, 
                                  anchor="nw")
        right_canvas.configure(yscrollcommand=right_scrollbar.set)
        
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.sidebet_widgets: Dict[str, Dict] = {}
    
    def show_past_games(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Pre-Game Historical Data")
        dialog.geometry("1400x850")
        dialog.configure(bg='#f5f5f5')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Header
        tk.Label(dialog, text="📊 HISTORICAL TRADING DATA", 
                font=('Arial', 20, 'bold'), bg='#f5f5f5', 
                fg='#0066cc').pack(pady=15)
        
        tk.Label(dialog, text="Review past game outcomes to inform your trading strategy", 
                font=('Arial', 14), bg='#f5f5f5', 
                fg='#666666').pack(pady=(0, 15))
        
        # Create scrollable frame for game grid
        canvas_frame = tk.Frame(dialog, bg='#f5f5f5')
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=15)
        
        canvas = tk.Canvas(canvas_frame, bg='#f5f5f5', highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        grid_frame = tk.Frame(canvas, bg='#f5f5f5')
        
        grid_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=grid_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Create 4x5 grid for 20 games
        for i, t in enumerate(self.past, 1):
            row = (i - 1) // 5
            col = (i - 1) % 5
            
            # Compact card
            card = tk.Frame(grid_frame, bg='#ffffff', relief=tk.RAISED, bd=2)
            card.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
            
            # Game number
            tk.Label(card, text=f"Game {i}", 
                    font=('Arial', 10, 'bold'), bg='#e6f2ff', 
                    fg='#0066cc').pack(fill=tk.X, pady=2)
            
            # Team A - compact
            tk.Label(card, text="Team A", 
                    font=('Arial', 9, 'bold'), bg='#ffffff', 
                    fg='#ff6600').pack(anchor=tk.W, padx=8, pady=(5,2))
            tk.Label(card, text=f"🍊 {t.oranges_A}  🍎 {t.lemons_A}", 
                    font=('Arial', 11, 'bold'), bg='#ffffff', 
                    fg='#000000').pack(anchor=tk.W, padx=8, pady=2)
            
            # Separator line
            tk.Frame(card, bg='#cccccc', height=1).pack(fill=tk.X, padx=8, pady=3)
            
            # Team B - compact
            tk.Label(card, text="Team B", 
                    font=('Arial', 9, 'bold'), bg='#ffffff', 
                    fg='#0066cc').pack(anchor=tk.W, padx=8, pady=2)
            tk.Label(card, text=f"🍊 {t.oranges_B}  🍎 {t.lemons_B}", 
                    font=('Arial', 11, 'bold'), bg='#ffffff', 
                    fg='#000000').pack(anchor=tk.W, padx=8, pady=(2,8))
        
        # Configure grid weights for equal sizing
        for i in range(5):
            grid_frame.columnconfigure(i, weight=1, uniform='col')
        for i in range(4):
            grid_frame.rowconfigure(i, weight=1, uniform='row')
        
        # Start button
        btn_frame = tk.Frame(dialog, bg='#f5f5f5')
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="▶️  START TRADING  ▶️", 
                 font=('Arial', 16, 'bold'), 
                 bg='#00cc00', fg='white', width=25, height=2,
                 activebackground='#009900', relief=tk.RAISED, bd=4,
                 cursor='hand2',
                 command=lambda: [dialog.destroy(), self.start_game()]).pack(pady=5)
    
    def start_game(self):
        """Initialize and start the trading game."""
        print("Starting game...")
        self.game_running = True
        self.engine.reset()
        self.engine.game_start_time = datetime.now()
        
        # Start background threads
        game_thread = threading.Thread(target=self.game_loop, daemon=True)
        display_thread = threading.Thread(target=self.display_update_thread, daemon=True)
        
        game_thread.start()
        display_thread.start()
        
        # Initial display update
        self.update_display()
    
    def game_loop(self):
        """Main game loop running in background thread."""
        try:
            while self.game_running and not self._shutdown_event.is_set():
                # Wait if paused
                while self.game_paused and self.game_running:
                    time.sleep(0.3)
                
                if not self.game_running:
                    break
                
                should_update_display = False
                should_show_knowledge = False
                should_show_quiz = False
                should_settle = False
                
                try:
                    with self.update_lock:
                        if self.engine.tick < self.cfg.ticks and not self.game_paused:
                            # Check if it's time for next tick
                            now = datetime.now()
                            if (self.engine.next_tick_time is None or 
                                now >= self.engine.next_tick_time):
                                
                                self.engine.simulate_tick()
                                self.engine.next_tick_time = (now + 
                                    timedelta(seconds=self.cfg.tick_interval_seconds))
                                
                                # Check expired side bets
                                self.engine.check_expired_sidebets()
                                
                                # Maybe spawn new side bet
                                self.engine.maybe_spawn_sidebet()
                                
                                should_update_display = True
                                
                                # Decide if we should trigger quizzes (only after tick 5)
                                if self.engine.tick > 5:
                                    if self.cfg.pause_for_questions:
                                        if random.random() < self.cfg.knowledge_check_probability:
                                            self.game_paused = True
                                            should_show_knowledge = True
                                    
                                    if self.cfg.market_quiz_enabled and not self.game_paused:
                                        if random.random() < self.cfg.market_quiz_probability:
                                            self.game_paused = True
                                            should_show_quiz = True
                        else:
                            # Game complete
                            self.game_running = False
                            should_settle = True
                
                except Exception as e:
                    print(f"Game loop error: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Schedule UI updates OUTSIDE the lock
                if should_update_display and self.game_running and not self._shutdown_event.is_set():
                    try:
                        self.root.after(0, self.update_display)
                    except (tk.TclError, RuntimeError):
                        pass
                
                if should_show_knowledge and self.game_running and not self._shutdown_event.is_set():
                    try:
                        self.root.after(100, self.show_knowledge_check)
                    except (tk.TclError, RuntimeError):
                        self.game_paused = False
                
                if should_show_quiz and self.game_running and not self._shutdown_event.is_set():
                    try:
                        self.root.after(100, self.show_market_quiz)
                    except (tk.TclError, RuntimeError):
                        self.game_paused = False
                
                if should_settle and not self._shutdown_event.is_set():
                    try:
                        self.root.after(0, self.show_settlement)
                    except (tk.TclError, RuntimeError):
                        pass
                    break
                
                # Sleep between checks
                time.sleep(0.2)
        
        except Exception as e:
            print(f"Fatal game thread error: {e}")
            import traceback
            traceback.print_exc()
            self.game_running = False
    
    def display_update_thread(self):
        """Continuously update the display every 500ms (slower for learning)"""
        while self.game_running and not self._shutdown_event.is_set():
            try:
                if self.game_running:  # Double check
                    self.root.after_idle(self.update_countdown)
                time.sleep(0.5)  # 500ms interval for slower, more learnable pace
            except Exception as e:
                print(f"Display update error: {e}")
                break
    
    def update_countdown(self):
        """Update game timer and side bet countdowns."""
        if not self.game_running:
            return
        
        now = datetime.now()
        
        # Update main game timer
        try:
            if self.engine.game_start_time:
                total_real_seconds = self.cfg.ticks * self.cfg.tick_interval_seconds
                elapsed_real = (now - self.engine.game_start_time).total_seconds()
                remaining_real = max(0, total_real_seconds - elapsed_real)
                
                # Convert to game time (6x multiplier)
                remaining_game_seconds = remaining_real * self.cfg.display_time_multiplier
                minutes = int(remaining_game_seconds // 60)
                seconds = int(remaining_game_seconds % 60)
                
                # Update UI
                try:
                    if self.game_running:
                        tick_value = self.engine.tick
                        self.progress_bar['value'] = tick_value
                        self.time_label.config(text=f"{minutes}:{seconds:02d}")
                except tk.TclError:
                    return
        except (AttributeError, TypeError, ValueError):
            pass
        
        # Update side bet countdown timers
        if not self.game_running:
            return
        
        # Get snapshot of side bet widgets
        try:
            widget_snapshot = list(self.sidebet_widgets.items())
        except RuntimeError:
            return
        
        for sid, widgets in widget_snapshot:
            if not self.game_running:
                break
            try:
                with self.update_lock:
                    sb = self.engine.sidebets.get(sid)
                if sb and 'countdown' in widgets:
                    remaining = (sb.expiry_time - now).total_seconds()
                    if remaining > 0:
                        try:
                            widgets['countdown'].config(text=f"⏱ {remaining:.0f}s")
                        except (tk.TclError, RuntimeError):
                            pass
                    else:
                        try:
                            widgets['countdown'].config(text="⏱ EXPIRED")
                        except (tk.TclError, RuntimeError):
                            pass
            except Exception:
                pass
    
    def update_display(self):
        """Update all UI elements with current game state - thread-safe."""
        if not self.game_running:
            return
        
        # Snapshot data inside lock
        try:
            with self.update_lock:
                # Copy all tally values atomically
                t_oranges_A = self.engine.tally.oranges_A
                t_lemons_A = self.engine.tally.lemons_A
                t_oranges_B = self.engine.tally.oranges_B
                t_lemons_B = self.engine.tally.lemons_B
                
                # Detect changes for flash effects
                changed_a_oranges = False
                changed_a_apples = False
                changed_b_oranges = False
                changed_b_apples = False
                
                if self.last_tally is not None:
                    try:
                        changed_a_oranges = t_oranges_A > self.last_tally.oranges_A
                        changed_a_apples = t_lemons_A > self.last_tally.lemons_A
                        changed_b_oranges = t_oranges_B > self.last_tally.oranges_B
                        changed_b_apples = t_lemons_B > self.last_tally.lemons_B
                    except (AttributeError, TypeError):
                        pass
                
                # Calculate fresh quotes (less frequent for main market)
                if self.engine.tick % 3 == 0:  # Update main price every 3 ticks
                    main_px, side_prices = self.engine.current_quotes()
                    self.last_main_px = main_px
                else:
                    _, side_prices = self.engine.current_quotes()
                    main_px = self.last_main_px
                
                self.last_side_px.update(side_prices)
                
                # Update last tally
                self.last_tally = self.engine.tally.copy()
                
                # Current value
                current_value = self.engine.settle_value()
                
                # Position
                main_position = self.engine.pos.main_qty
                
                # Get current side bets
                current_sids = set(self.engine.sidebets.keys())
        
        except Exception as e:
            print(f"Data snapshot error: {e}")
            return
        
        # Update UI elements (outside lock)
        try:
            # Update fruit counts with flash effects
            if self.game_running:
                # Team A
                color_a_o = '#ff9900' if changed_a_oranges else '#000000'
                color_a_a = '#ff9900' if changed_a_apples else '#000000'
                
                self.oranges_a_label.config(text=f"🍊 {t_oranges_A}", fg=color_a_o)
                self.lemons_a_label.config(text=f"🍎 {t_lemons_A}", fg=color_a_a)
                
                # Team B
                color_b_o = '#ff9900' if changed_b_oranges else '#000000'
                color_b_a = '#ff9900' if changed_b_apples else '#000000'
                
                self.oranges_b_label.config(text=f"🍊 {t_oranges_B}", fg=color_b_o)
                self.lemons_b_label.config(text=f"🍎 {t_lemons_B}", fg=color_b_a)
                
                # Current value and price
                self.current_value_label.config(text=f"{current_value:,}")
                self.main_price_label.config(text=f"{main_px:,}")
                
                # Position
                pos_color = '#009900' if main_position > 0 else '#cc0000' if main_position < 0 else '#000000'
                self.position_label.config(text=f"{main_position:+d}", fg=pos_color)
                
        except Exception as e:
            print(f"UI main update error: {e}")
        
        # Update side bets
        if self.game_running:
            try:
                displayed_sids = set(self.sidebet_widgets.keys())
                
                # Remove widgets for expired side bets
                for sid in (displayed_sids - current_sids):
                    self.remove_sidebet_widget(sid)
                
                # Add widgets for new side bets
                for sid in (current_sids - displayed_sids):
                    self.add_sidebet_widget(sid)
                
                # Update prices for existing side bets
                for sid in current_sids:
                    if sid in self.sidebet_widgets:
                        try:
                            px = side_prices.get(sid, 0)
                            if px > 0 and 'price' in self.sidebet_widgets[sid]:
                                self.sidebet_widgets[sid]['price'].config(
                                    text=f"Price: {px:,}")
                        except (KeyError, tk.TclError, RuntimeError):
                            pass
            
            except Exception as e:
                print(f"Side bet update error: {e}")
        
        # Reset flash colors after brief delay
        if self.game_running:
            self.root.after(300, self.reset_flash_colors)
    
    def reset_flash_colors(self):
        """Reset flash colors back to normal."""
        try:
            if self.game_running:
                self.oranges_a_label.config(fg='#000000')
                self.lemons_a_label.config(fg='#000000')
                self.oranges_b_label.config(fg='#000000')
                self.lemons_b_label.config(fg='#000000')
        except tk.TclError:
            pass
    
    def add_sidebet_widget(self, sid: str):
        """Create and display a side bet widget."""
        if not self.game_running:
            return
        
        # Fetch side bet data with lock
        try:
            with self.update_lock:
                sb = self.engine.sidebets.get(sid)
                if not sb:
                    return
                
                # Ensure price is calculated
                if sb.price == 0:
                    try:
                        _, _ = self.engine.current_quotes()
                        sb = self.engine.sidebets.get(sid)
                        if not sb or sb.price == 0:
                            return
                    except:
                        return
                
                # Copy data for UI creation
                description = sb.description
                price = sb.price
        except Exception as e:
            print(f"Error fetching side bet {sid}: {e}")
            return
        
        # Create widgets (outside lock)
        try:
            # Alternate between left and right panels
            num_widgets = len(self.sidebet_widgets)
            container = (self.left_sidebets_container if num_widgets % 2 == 0 
                        else self.right_sidebets_container)
            
            frame = tk.Frame(container, bg='#f0f0f0', relief=tk.RAISED, bd=2)
            frame.pack(fill=tk.X, pady=3, padx=3)
            
            header = tk.Frame(frame, bg='#f0f0f0')
            header.pack(fill=tk.X, padx=5, pady=5)
            
            desc_label = tk.Label(header, text=description, 
                                 font=('Arial', 12, 'bold'), 
                                 bg='#f0f0f0', fg='#8800cc', 
                                 wraplength=220, justify=tk.LEFT)
            desc_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            countdown = tk.Label(header, text="⏱ 12s", 
                               font=('Arial', 11, 'bold'), 
                               bg='#f0f0f0', fg='#0066cc')
            countdown.pack(side=tk.RIGHT)
            
            price_label = tk.Label(frame, text=f"Price: {price:,}", 
                                  font=('Arial', 14, 'bold'), 
                                  bg='#f0f0f0', fg='#cc0000')
            price_label.pack(pady=5)
            
            controls = tk.Frame(frame, bg='#f0f0f0')
            controls.pack(pady=5)
            
            qty_frame = tk.Frame(controls, bg='#f0f0f0')
            qty_frame.pack(side=tk.TOP, pady=2)
            
            tk.Label(qty_frame, text="Qty:", font=('Arial', 10), 
                    bg='#f0f0f0').pack(side=tk.LEFT)
            
            qty_entry = tk.Entry(qty_frame, width=5, font=('Arial', 10), 
                                justify='center')
            qty_entry.pack(side=tk.LEFT, padx=2)
            qty_entry.insert(0, "1")
            
            btn_frame = tk.Frame(controls, bg='#f0f0f0')
            btn_frame.pack(side=tk.TOP, pady=2)
            
            buy_btn = tk.Button(btn_frame, text="BUY", 
                               font=('Arial', 10, 'bold'), 
                               bg='#00cc00', fg='black', width=6,
                               command=lambda: self.execute_trade('buy', sid))
            buy_btn.pack(side=tk.LEFT, padx=2)
            
            sell_btn = tk.Button(btn_frame, text="SELL", 
                                font=('Arial', 10, 'bold'), 
                                bg='#ff4400', fg='black', width=6,
                                command=lambda: self.execute_trade('sell', sid))
            sell_btn.pack(side=tk.LEFT, padx=2)
            
            # Store widget references
            self.sidebet_widgets[sid] = {
                'frame': frame,
                'price': price_label,
                'qty_entry': qty_entry,
                'countdown': countdown
            }
            
        except Exception as e:
            print(f"Error creating side bet widget {sid}: {e}")
    
    def remove_sidebet_widget(self, sid: str):
        """Remove a side bet widget."""
        try:
            if sid in self.sidebet_widgets:
                widgets = self.sidebet_widgets.pop(sid)
                if 'frame' in widgets:
                    widgets['frame'].destroy()
        except Exception as e:
            print(f"Error removing side bet widget {sid}: {e}")
    
    def execute_trade(self, side: str, instrument: str):
        """Execute a trade."""
        try:
            if instrument == 'main':
                qty_str = self.main_qty_entry.get().strip()
            else:
                if instrument in self.sidebet_widgets:
                    qty_str = self.sidebet_widgets[instrument]['qty_entry'].get().strip()
                else:
                    return
            
            qty = int(qty_str)
            if qty <= 0:
                raise ValueError("Quantity must be positive")
            
            if side == 'sell':
                qty = -qty
            
            with self.update_lock:
                if instrument == 'main':
                    price = self.last_main_px
                    self.engine.pos.main_qty += qty
                    self.engine.pos.main_trades.append((qty, price))
                    
                    self.status_message = f"{'BOUGHT' if qty > 0 else 'SOLD'} {abs(qty)} MAIN @ {price:,}"
                else:
                    if instrument in self.last_side_px:
                        price = self.last_side_px[instrument]
                        if instrument not in self.engine.pos.side_qties:
                            self.engine.pos.side_qties[instrument] = 0
                        self.engine.pos.side_qties[instrument] += qty
                        
                        if instrument not in self.engine.pos.side_trades:
                            self.engine.pos.side_trades[instrument] = []
                        self.engine.pos.side_trades[instrument].append((qty, price))
                        
                        sb = self.engine.sidebets.get(instrument)
                        desc = sb.description if sb else instrument
                        self.status_message = f"{'BOUGHT' if qty > 0 else 'SOLD'} {abs(qty)} {desc} @ {price:,}"
            
            # Update status bar
            if self.status_message:
                self.status_bar.config(text=self.status_message, bg='#e8f5e8', fg='#006600')
                # Clear message after 3 seconds
                self.root.after(3000, lambda: self.status_bar.config(
                    text="Ready to trade", bg='#e8f5e8', fg='#006600'))
        
        except (ValueError, tk.TclError) as e:
            self.status_bar.config(text=f"Trade Error: {e}", bg='#ffe8e8', fg='#cc0000')
            self.root.after(3000, lambda: self.status_bar.config(
                text="Ready to trade", bg='#e8f5e8', fg='#006600'))
        except Exception as e:
            print(f"Trade execution error: {e}")
    
    def _calculate_current_value(self) -> int:
        """Calculate current mark-to-market value of all positions."""
        main_px, side_prices = self.engine.current_quotes()
        value = self.engine.pos.main_qty * main_px
        for sid, qty in self.engine.pos.side_qties.items():
            px = side_prices.get(sid, 0)
            value += qty * px
        return int(value)
    
    def _calculate_cost_basis(self) -> int:
        """Calculate total cost basis of all positions."""
        cost = 0
        for qty, price in self.engine.pos.main_trades:
            cost += qty * price
        for sid, trades in self.engine.pos.side_trades.items():
            for qty, price in trades:
                cost += qty * price
        return int(cost)
    
    def _compute_minmax_pnl(self) -> Tuple[int, int]:
        """Compute approximate min and max P&L."""
        p = self.engine.pos
        cfg = self.cfg
        o_candidates = [cfg.oranges_min, cfg.oranges_max]
        l_candidates = [cfg.lemons_min, cfg.lemons_max]
        
        main_payoffs = [o * l for o in o_candidates for l in l_candidates]
        main_min, main_max = min(main_payoffs), max(main_payoffs)
        
        # Calculate VWAP for main
        if self.engine.pos.main_trades:
            total_qty = sum(q for q, _ in self.engine.pos.main_trades)
            if total_qty != 0:
                vwap = sum(q * p for q, p in self.engine.pos.main_trades) / total_qty
                pnl_main_min = int(p.main_qty * (main_min - vwap))
                pnl_main_max = int(p.main_qty * (main_max - vwap))
            else:
                pnl_main_min = pnl_main_max = 0
        else:
            pnl_main_min = pnl_main_max = 0
        
        # Simplified side bet bounds
        side_min = side_max = 0
        
        return int(pnl_main_min + side_min), int(pnl_main_max + side_max)
    
    def show_knowledge_check(self):
        """Pause game and ask knowledge check questions."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Knowledge Check - Game Paused")
        dialog.geometry("800x700")
        dialog.configure(bg='#ffffff')
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="⏸️ KNOWLEDGE CHECK ⏸️", 
                font=('Arial', 18, 'bold'), bg='#ffffff', 
                fg='#ff0000').pack(pady=20)
        
        tk.Label(dialog, text="Game paused. Answer these questions:", 
                font=('Arial', 12), bg='#ffffff', fg='#000000').pack(pady=10)
        
        # Scrollable question frame
        canvas = tk.Canvas(dialog, bg='#ffffff')
        scrollbar = tk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#ffffff')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=20)
        scrollbar.pack(side="right", fill="y")
        
        questions = []
        
        # Question 1: Current position
        with self.update_lock:
            pos_str = self.engine.pos.net_exposure_str()
        questions.append(("1. What is your current net position?", 
                         f"Answer: {pos_str}"))
        
        # Question 2: Estimated P&L
        if self.engine.pos.main_qty != 0 or len(self.engine.pos.side_qties) > 0:
            current_value = self._calculate_current_value()
            cost_basis = self._calculate_cost_basis()
            unrealized_pnl = current_value - cost_basis
            questions.append(("2. Estimate your unrealized P&L (within 20%)", 
                             f"Actual P&L: ${unrealized_pnl:,}"))
        
        # Question 3: Max/Min P&L
        min_pnl, max_pnl = self._compute_minmax_pnl()
        questions.append(("3. What's your estimated MAXIMUM P&L?", 
                         f"Approximate max: ${max_pnl:,}"))
        questions.append(("4. What's your estimated MINIMUM P&L?", 
                         f"Approximate min: ${min_pnl:,}"))
        
        # Question 5: Last trade analysis
        if self.last_trade_info:
            trade = self.last_trade_info
            if trade['type'] == 'main':
                q = (f"5. Your last trade: {trade['side'].upper()} MAIN "
                     f"x{trade['qty']} @ {trade['price']}. "
                     f"What factors influenced this decision?")
                a = "Consider: current counts, expected values, risk/reward"
                questions.append((q, a))
            else:
                q = (f"5. Your last trade: {trade['side'].upper()} "
                     f"{trade['instrument']} x{trade['qty']} @ {trade['price']}. "
                     f"Do you think this was +EV?")
                a = f"Consider: {trade['description']} payoff vs premium paid"
                questions.append((q, a))
        
        # Display questions
        for i, (question, answer) in enumerate(questions):
            q_frame = tk.Frame(scrollable_frame, bg='#f0f0f0', 
                              relief=tk.SUNKEN, bd=2)
            q_frame.pack(fill=tk.X, pady=10, padx=10)
            
            q_label = tk.Label(q_frame, text=question, font=('Arial', 11, 'bold'), 
                              bg='#f0f0f0', fg='#000000', wraplength=700, 
                              justify=tk.LEFT)
            q_label.pack(anchor=tk.W, padx=10, pady=10)
            
            # Input box for answer
            answer_entry = tk.Entry(q_frame, font=('Arial', 10), width=60)
            answer_entry.pack(padx=20, pady=5)
            
            # Show answer button
            def show_ans(ans=answer, entry=answer_entry):
                entry.delete(0, tk.END)
                entry.insert(0, ans)
                entry.config(state='readonly', fg='#0066cc')
            
            tk.Button(q_frame, text="Show Answer", font=('Arial', 10), 
                     bg='#ffcc00', fg='#000000', 
                     command=show_ans).pack(padx=20, pady=10)
        
        def resume_game():
            self.game_paused = False
            dialog.destroy()
        
        # Resume button at bottom of dialog (outside canvas)
        btn_frame = tk.Frame(dialog, bg='#ffffff')
        btn_frame.pack(side='bottom', pady=20)
        tk.Button(btn_frame, text="▶️ RESUME GAME", font=('Arial', 14, 'bold'), 
                 bg='#00ff00', fg='#000000', width=20,
                 command=resume_game).pack()
    
    def show_market_quiz(self):
        """Pause game and quiz user on market opportunity."""
        # This will be added in next step
        self.game_paused = False
    
    def show_settlement(self):
        """Show final settlement dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Game Complete - Final Settlement")
        dialog.geometry("800x600")
        dialog.configure(bg='#f0f8ff')
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="🏁 GAME COMPLETE!", 
                font=('Arial', 24, 'bold'), bg='#f0f8ff', 
                fg='#0066cc').pack(pady=20)
        
        # Final tally
        t = self.engine.tally
        final_value = self.engine.settle_value()
        
        info_frame = tk.Frame(dialog, bg='#ffffff', relief=tk.RAISED, bd=2)
        info_frame.pack(pady=20, padx=20, fill=tk.X)
        
        tk.Label(info_frame, text="Final Game State", 
                font=('Arial', 18, 'bold'), bg='#ffffff', 
                fg='#0066cc').pack(pady=10)
        
        tk.Label(info_frame, text=f"Total Oranges: {t.oranges_total}", 
                font=('Arial', 14), bg='#ffffff').pack(pady=2)
        tk.Label(info_frame, text=f"Total Lemons: {t.lemons_total}", 
                font=('Arial', 14), bg='#ffffff').pack(pady=2)
        tk.Label(info_frame, text=f"Final Payoff: {final_value:,}", 
                font=('Arial', 16, 'bold'), bg='#ffffff', 
                fg='#cc0000').pack(pady=10)
        
        # Settlement
        settlement_frame = tk.Frame(dialog, bg='#ffffff', relief=tk.RAISED, bd=2)
        settlement_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        tk.Label(settlement_frame, text="Your P&L Settlement", 
                font=('Arial', 18, 'bold'), bg='#ffffff', 
                fg='#0066cc').pack(pady=10)
        
        settlement_text = scrolledtext.ScrolledText(settlement_frame, height=15, 
                                                   font=('Monaco', 11))
        settlement_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Calculate P&L
        total_pnl = 0
        settlement_lines = []
        
        # Main market P&L
        main_qty = self.engine.pos.main_qty
        if main_qty != 0:
            main_pnl = main_qty * final_value
            # Subtract cost basis
            cost_basis = sum(qty * price for qty, price in self.engine.pos.main_trades)
            main_net_pnl = main_pnl - cost_basis
            total_pnl += main_net_pnl
            
            settlement_lines.append(f"MAIN MARKET:")
            settlement_lines.append(f"  Position: {main_qty:+d} contracts")
            settlement_lines.append(f"  Final Value: {final_value:,}")
            settlement_lines.append(f"  Gross P&L: {main_pnl:+,}")
            settlement_lines.append(f"  Cost Basis: {cost_basis:+,}")
            settlement_lines.append(f"  Net P&L: {main_net_pnl:+,}")
            settlement_lines.append("")
        
        # Side bet P&L
        for sid, qty in self.engine.pos.side_qties.items():
            if qty != 0:
                sb = self.engine.get_sidebet(sid)
                if sb:
                    side_payoff = self.engine.side_payoff(sb)
                    side_pnl = qty * side_payoff
                    
                    # Cost basis for this side bet
                    if sid in self.engine.pos.side_trades:
                        side_cost = sum(q * p for q, p in self.engine.pos.side_trades[sid])
                    else:
                        side_cost = 0
                    
                    side_net_pnl = side_pnl - side_cost
                    total_pnl += side_net_pnl
                    
                    settlement_lines.append(f"SIDE BET {sid} ({sb.description}):")
                    settlement_lines.append(f"  Position: {qty:+d} contracts")
                    settlement_lines.append(f"  Final Payoff: {side_payoff:,}")
                    settlement_lines.append(f"  Gross P&L: {side_pnl:+,}")
                    settlement_lines.append(f"  Cost Basis: {side_cost:+,}")
                    settlement_lines.append(f"  Net P&L: {side_net_pnl:+,}")
                    settlement_lines.append("")
        
        settlement_lines.append("=" * 50)
        settlement_lines.append(f"TOTAL NET P&L: {total_pnl:+,}")
        
        settlement_text.insert(tk.END, "\n".join(settlement_lines))
        settlement_text.config(state=tk.DISABLED)
        
        # Close button
        tk.Button(dialog, text="Close", font=('Arial', 14, 'bold'), 
                 bg='#cccccc', width=15,
                 command=dialog.destroy).pack(pady=20)
    
    def on_closing(self):
        """Handle window close event."""
        print("Window closing...")
        self.game_running = False
        self._shutdown_event.set()
        time.sleep(0.5)  # Give threads time to shutdown
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass


def main():
    print("Starting Optiver Trading Simulator...")
    
    try:
        cfg = GameConfig()
        root = tk.Tk()
        
        # Set up window close handler
        app = TradingGUI(root, cfg)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        print("GUI initialized successfully")
        print("Starting main loop...")
        
        root.mainloop()
        
        print("Main loop ended")
        
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Application closed successfully")


if __name__ == "__main__":
    main()