#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optiver-style Fruit Trading Simulator (GUI Version)

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


# =============================
# Configuration
# =============================
@dataclass
class GameConfig:
    ticks: int = 36  # 3 minutes total (36 ticks × 5s)

    # Poisson means per tick (TOTAL across both teams)
    # Spread over 36 ticks to give E[total oranges] ≈ 16 and E[total apples] ≈ 13
    # resulting in E[product] ≈ 208 with good variance
    lambda_oranges: float = 0.44
    lambda_lemons: float = 0.36

    # Team strengths (probability of assigning an event to Team A)
    team_A_strength_orange: float = 0.52
    team_A_strength_lemon: float = 0.48

    # Bounds for quick risk panel
    oranges_min: int = 8
    oranges_max: int = 28
    lemons_min: int = 6
    lemons_max: int = 22

    # Quote noise parameters (applied to theoretical EV, then rounded to int)
    sigma_normal: float = 0.07
    sigma_shock: float = 0.25
    shock_prob: float = 0.08

    # Side bets
    sidebet_prob_per_spawn: float = 0.85  # 85% chance
    max_concurrent_sidebets: int = 8  # More concurrent bets

    # Past games to display (no aggregates shown)
    past_games: int = 20
    # Real-time behaviour
    tick_interval_seconds: int = 5
    sidebet_duration_seconds: int = 12  # Slightly longer base duration
    sidebet_duration_variance: int = 6  # +/- 6s variance (6-18s range)
    sidebet_spawn_interval_seconds: int = 3  # Check every 3 seconds
    
    # Knowledge check pauses
    knowledge_check_probability: float = 0.08  # 8% chance per tick (~3 pauses per game)
    pause_for_questions: bool = True

    seed: Optional[int] = None


# =============================
# Model types
# =============================
@dataclass
class Tally:
    oranges_A: int = 0
    oranges_B: int = 0
    lemons_A: int = 0
    lemons_B: int = 0

    @property
    def oranges_total(self) -> int:
        return self.oranges_A + self.oranges_B

    @property
    def lemons_total(self) -> int:
        return self.lemons_A + self.lemons_B

    def copy(self) -> 'Tally':
        return Tally(self.oranges_A, self.oranges_B, self.lemons_A, self.lemons_B)


# Side bet function and types
SideBetFn = Callable[[Tally], int]


@dataclass
class SideBet:
    sid: str
    kind: str
    description: str
    payoff: SideBetFn
    price: int
    expiry_time: datetime
    spawn_time: datetime


@dataclass
class Position:
    main_qty: int = 0
    side_qties: Dict[str, int] = None
    main_trades: List[Tuple[int, int]] = None
    side_trades: Dict[str, List[Tuple[int, int]]] = None

    def __post_init__(self):
        if self.side_qties is None:
            self.side_qties = {}
        if self.main_trades is None:
            self.main_trades = []
        if self.side_trades is None:
            self.side_trades = {}

    def net_exposure_str(self) -> str:
        parts = [f"MAIN: {self.main_qty}"]
        for sid, q in sorted(self.side_qties.items()):
            parts.append(f"{sid}: {q}")
        return ", ".join(parts)


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
    l = current.lemons_total
    return o * l + o * mu_lemons + l * mu_oranges + mu_oranges * mu_lemons


def ev_2_pow_oranges_A(current: Tally, mu_future_A_oranges: float) -> float:
    return (2 ** current.oranges_A) * math.exp(mu_future_A_oranges)


def ev_sum_oranges_lemons(current: Tally, mu_oranges: float, mu_lemons: float) -> float:
    return current.oranges_total + current.lemons_total + mu_oranges + mu_lemons


def ev_oranges_A_linear(current: Tally, mu_future_A_oranges: float) -> float:
    return current.oranges_A + mu_future_A_oranges


def ev_product_oranges_A_apples_A(current: Tally, mu_future_A_oranges: float,
                                   mu_future_A_lemons: float) -> float:
    return (current.oranges_A * current.lemons_A + 
            current.oranges_A * mu_future_A_lemons +
            current.lemons_A * mu_future_A_oranges +
            mu_future_A_oranges * mu_future_A_lemons)


def ev_sum_team_A(current: Tally, mu_future_A_oranges: float,
                  mu_future_A_lemons: float) -> float:
    return (current.oranges_A + current.lemons_A + 
            mu_future_A_oranges + mu_future_A_lemons)


def ev_sum_team_B(current: Tally, mu_future_B_oranges: float,
                  mu_future_B_lemons: float) -> float:
    return (current.oranges_B + current.lemons_B + 
            mu_future_B_oranges + mu_future_B_lemons)


def ev_product_oranges_B_apples_B(current: Tally, mu_future_B_oranges: float,
                                   mu_future_B_lemons: float) -> float:
    return (current.oranges_B * current.lemons_B + 
            current.oranges_B * mu_future_B_lemons +
            current.lemons_B * mu_future_B_oranges +
            mu_future_B_oranges * mu_future_B_lemons)


def ev_oranges_squared(current: Tally, mu_oranges: float) -> float:
    o = current.oranges_total
    return o * o + 2 * o * mu_oranges + mu_oranges * mu_oranges


def ev_apples_squared(current: Tally, mu_lemons: float) -> float:
    l = current.lemons_total
    return l * l + 2 * l * mu_lemons + mu_lemons * mu_lemons


def ev_oranges_A_squared(current: Tally, mu_future_A_oranges: float) -> float:
    o = current.oranges_A
    return o * o + 2 * o * mu_future_A_oranges + mu_future_A_oranges ** 2


def ev_difference_oranges_apples(current: Tally, mu_oranges: float, 
                                  mu_lemons: float) -> float:
    return abs((current.oranges_total + mu_oranges) - 
               (current.lemons_total + mu_lemons))


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

    def check_expired_sidebets(self):
        now = datetime.now()
        for sb in list(self.sidebets.values()):
            if now >= sb.expiry_time:
                self.archive_sidebets[sb.sid] = sb
                del self.sidebets[sb.sid]

    def current_quotes(self) -> Tuple[int, Dict[str, int]]:
        main_ev = main_ev_product(self.tally, self.remaining_mu_oranges, 
                                  self.remaining_mu_lemons)
        main_px = _noisy_int_quote(main_ev, self.cfg)
        side_prices: Dict[str, int] = {}
        for sid, sb in self.sidebets.items():
            mu_A_oranges = (self.cfg.team_A_strength_orange * 
                           self.remaining_mu_oranges)
            mu_A_lemons = (self.cfg.team_A_strength_lemon * 
                          self.remaining_mu_lemons)
            mu_B_oranges = ((1 - self.cfg.team_A_strength_orange) * 
                           self.remaining_mu_oranges)
            mu_B_lemons = ((1 - self.cfg.team_A_strength_lemon) * 
                          self.remaining_mu_lemons)
            
            if sb.kind == 'pow2_oranges_A':
                ev = ev_2_pow_oranges_A(self.tally, mu_A_oranges)
            elif sb.kind == 'sum_O_L' or sb.kind == 'sum_all':
                ev = ev_sum_oranges_lemons(self.tally, self.remaining_mu_oranges, 
                                          self.remaining_mu_lemons)
            elif sb.kind == 'oranges_A_linear':
                ev = ev_oranges_A_linear(self.tally, mu_A_oranges)
            elif sb.kind == 'product_OA_AA':
                ev = ev_product_oranges_A_apples_A(self.tally, mu_A_oranges, 
                                                   mu_A_lemons)
            elif sb.kind == 'sum_team_A':
                ev = ev_sum_team_A(self.tally, mu_A_oranges, mu_A_lemons)
            elif sb.kind == 'sum_team_B':
                ev = ev_sum_team_B(self.tally, mu_B_oranges, mu_B_lemons)
            elif sb.kind == 'product_OB_AB':
                ev = ev_product_oranges_B_apples_B(self.tally, mu_B_oranges, 
                                                   mu_B_lemons)
            elif sb.kind == 'oranges_squared':
                ev = ev_oranges_squared(self.tally, self.remaining_mu_oranges)
            elif sb.kind == 'apples_squared':
                ev = ev_apples_squared(self.tally, self.remaining_mu_lemons)
            elif sb.kind == 'oranges_A_squared':
                ev = ev_oranges_A_squared(self.tally, mu_A_oranges)
            elif sb.kind == 'diff_O_A':
                ev = ev_difference_oranges_apples(self.tally, 
                                                  self.remaining_mu_oranges,
                                                  self.remaining_mu_lemons)
            elif sb.kind == 'oranges_B':
                ev = self.tally.oranges_B + mu_B_oranges
            elif sb.kind == 'apples_A':
                ev = self.tally.lemons_A + mu_A_lemons
            else:
                ev = 0.0
            sb.price = _noisy_int_quote(ev, self.cfg)
            side_prices[sid] = sb.price
        return main_px, side_prices

    def side_payoff(self, sb: SideBet, tally: Optional[Tally] = None) -> int:
        t = tally or self.tally
        return int(sb.payoff(t))

    def settle_value(self, tally: Optional[Tally] = None) -> int:
        t = tally or self.tally
        return int(t.oranges_total * t.lemons_total)

    def maybe_spawn_sidebet(self):
        now = datetime.now()
        
        if self.last_sidebet_spawn_time is not None:
            time_since_last = (now - self.last_sidebet_spawn_time).total_seconds()
            if time_since_last < self.cfg.sidebet_spawn_interval_seconds:
                return
        
        if len(self.sidebets) >= self.cfg.max_concurrent_sidebets:
            return
        if random.random() > self.cfg.sidebet_prob_per_spawn:
            return
        
        choices: List[Tuple[str, str, SideBetFn]] = [
            ('pow2_oranges_A', '2^(Team A 🍊)', lambda t: 2 ** t.oranges_A),
            ('sum_O_L', 'Total 🍊 + Total 🍎', 
             lambda t: t.oranges_total + t.lemons_total),
            ('oranges_A_linear', 'Team A 🍊', lambda t: t.oranges_A),
            ('product_OA_AA', 'Team A: 🍊 × 🍎',
             lambda t: t.oranges_A * t.lemons_A),
            ('sum_team_A', 'Team A Total (🍊+🍎)',
             lambda t: t.oranges_A + t.lemons_A),
            ('sum_team_B', 'Team B Total (🍊+🍎)',
             lambda t: t.oranges_B + t.lemons_B),
            ('product_OB_AB', 'Team B: 🍊 × 🍎',
             lambda t: t.oranges_B * t.lemons_B),
            ('oranges_squared', '(Total 🍊)²',
             lambda t: t.oranges_total ** 2),
            ('apples_squared', '(Total 🍎)²',
             lambda t: t.lemons_total ** 2),
            ('oranges_A_squared', '(Team A 🍊)²',
             lambda t: t.oranges_A ** 2),
            ('diff_O_A', '|Total 🍊 - Total 🍎|',
             lambda t: abs(t.oranges_total - t.lemons_total)),
            ('sum_all', 'All Fruit (🍊+🍎)',
             lambda t: t.oranges_total + t.lemons_total),
            ('oranges_B', 'Team B 🍊',
             lambda t: t.oranges_B),
            ('apples_A', 'Team A 🍎',
             lambda t: t.lemons_A),
        ]
        kind, desc, fn = random.choice(choices)
        sid = f"SB{self.next_sb_id}"
        self.next_sb_id += 1
        
        spawn_time = now
        # Variable expiry time
        base_duration = self.cfg.sidebet_duration_seconds
        variance = random.randint(-self.cfg.sidebet_duration_variance,
                                 self.cfg.sidebet_duration_variance)
        duration = max(5, base_duration + variance)
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
# GUI Application
# =============================
class TradingGUI:
    def __init__(self, root: tk.Tk, cfg: GameConfig):
        self.root = root
        self.cfg = cfg
        self.engine = FruitEngine(cfg)
        self.past = simulate_past_games(cfg, cfg.past_games)
        
        self.game_running = False
        self.update_lock = threading.Lock()
        self.last_main_px = 0
        self.last_side_px: Dict[str, int] = {}
        self.game_paused = False
        self.last_trade_info = None  # Track last trade for questions
        self.status_message = None  # For status bar
        self.last_tally = Tally(0, 0, 0, 0)  # Track previous tally for flash detection
        
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
        
        # Countdown timer in seconds
        self.time_label = tk.Label(progress_frame, text="180s", 
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
        
        # Left side: Side bets (left panel) - FIXED HEIGHT
        left_sidebets_frame = tk.LabelFrame(center_wrapper, text="SIDE BETS", 
                                           font=('Arial', 12, 'bold'), 
                                           bg='#ffffff', fg='#8800cc', 
                                           relief=tk.RIDGE, bd=2, 
                                           width=300, height=500)
        left_sidebets_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, 
                                padx=(0, 5))
        left_sidebets_frame.pack_propagate(False)
        
        # Scrollable canvas for side bets
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
        
        # Center: Main game area
        center_frame = tk.Frame(center_wrapper, bg='#ffffff')
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Middle: Game State (Team Data)
        state_frame = tk.Frame(center_frame, bg='#ffffff')
        state_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left: Team A
        team_a_frame = tk.LabelFrame(state_frame, text="TEAM A", 
                                     font=('Arial', 14, 'bold'), bg='#ffffff', 
                                     fg='#cc0000', relief=tk.RIDGE, bd=3)
        team_a_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.team_a_oranges = tk.Label(team_a_frame, text="🍊 Oranges: 0", 
                                       font=('Arial', 24, 'bold'), bg='#ffffff', fg='#000000')
        self.team_a_oranges.pack(pady=25)
        
        self.team_a_apples = tk.Label(team_a_frame, text="🍎 Apples: 0", 
                                      font=('Arial', 24, 'bold'), bg='#ffffff', fg='#000000')
        self.team_a_apples.pack(pady=25)
        
        # Right: Team B
        team_b_frame = tk.LabelFrame(state_frame, text="TEAM B", 
                                     font=('Arial', 14, 'bold'), bg='#ffffff', 
                                     fg='#0066cc', relief=tk.RIDGE, bd=3)
        team_b_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.team_b_oranges = tk.Label(team_b_frame, text="🍊 Oranges: 0", 
                                       font=('Arial', 24, 'bold'), bg='#ffffff', fg='#000000')
        self.team_b_oranges.pack(pady=25)
        
        self.team_b_apples = tk.Label(team_b_frame, text="🍎 Apples: 0", 
                                      font=('Arial', 24, 'bold'), bg='#ffffff', fg='#000000')
        self.team_b_apples.pack(pady=25)
        
        # Bottom: Trading Area - FIXED HEIGHT
        trading_frame = tk.Frame(main_frame, bg='#ffffff', height=160)
        trading_frame.pack(fill=tk.X, expand=False, pady=(10, 0))
        trading_frame.pack_propagate(False)
        
        # Main Product Trading
        main_trade_frame = tk.LabelFrame(trading_frame, text="MAIN PRODUCT (O × A)", 
                                        font=('Arial', 14, 'bold'), bg='#ffffff', 
                                        fg='#ff8800', relief=tk.RIDGE, bd=3)
        main_trade_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.main_price_label = tk.Label(main_trade_frame, text="Price: ---", 
                                         font=('Arial', 26, 'bold'), bg='#ffffff', 
                                         fg='#ff8800')
        self.main_price_label.pack(pady=8)
        
        main_controls = tk.Frame(main_trade_frame, bg='#ffffff')
        main_controls.pack(pady=8)
        
        tk.Label(main_controls, text="Quantity:", font=('Arial', 14, 'bold'), 
                bg='#ffffff', fg='#000000').pack(side=tk.LEFT, padx=5)
        
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
        
        # Right side: More Side bets (right panel) - FIXED HEIGHT
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
        
        # Title section
        title_frame = tk.Frame(dialog, bg='#0066cc', height=70)
        title_frame.pack(fill=tk.X, pady=0)
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="📊 HISTORICAL MARKET DATA", 
                font=('Arial', 22, 'bold'), bg='#0066cc', 
                fg='#ffffff').pack(pady=20)
        
        # Subtitle
        tk.Label(dialog, text="Study these past 20 games carefully. Calculate expected values.", 
                font=('Arial', 12), bg='#f5f5f5', 
                fg='#333333').pack(pady=10)
        
        # Grid container (no scrolling)
        grid_frame = tk.Frame(dialog, bg='#f5f5f5')
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
                 command=lambda: [dialog.destroy(), self.start_game()]).pack()
        
    def start_game(self):
        self.game_running = True
        self.engine.game_start_time = datetime.now()
        self.engine.next_tick_time = (self.engine.game_start_time + 
                                     timedelta(seconds=self.cfg.tick_interval_seconds))
        
        # Initialize last_tally to current state
        self.last_tally = self.engine.tally.copy()
        
        # Initialize quotes before starting
        main_px, side_prices = self.engine.current_quotes()
        self.last_main_px = main_px
        self.last_side_px = dict(side_prices)
        
        # Start background threads
        threading.Thread(target=self.game_timer_thread, daemon=True).start()
        threading.Thread(target=self.display_update_thread, daemon=True).start()
        
        self.update_display()
        
    def game_timer_thread(self):
        while self.game_running:
            # Wait if paused
            while self.game_paused and self.game_running:
                time.sleep(0.5)
            
            if not self.game_running:
                break
                
            now = datetime.now()
            
            with self.update_lock:
                # Check and remove expired side bets immediately
                self.engine.check_expired_sidebets()
                
                if self.engine.next_tick_time and now >= self.engine.next_tick_time:
                    if self.engine.tick < self.cfg.ticks:
                        self.engine.simulate_tick()
                        self.engine.next_tick_time = now + timedelta(
                            seconds=self.cfg.tick_interval_seconds)
                        self.root.after(0, self.update_display)
                        
                        # Maybe pause for knowledge check
                        if self.cfg.pause_for_questions:
                            import random
                            if random.random() < self.cfg.knowledge_check_probability:
                                self.game_paused = True
                                self.root.after(100, self.show_knowledge_check)
                    else:
                        self.game_running = False
                        self.root.after(0, self.show_settlement)
                        break
                
                self.engine.maybe_spawn_sidebet()
            
            time.sleep(0.2)  # Check more frequently (5 times per second)
    
    def display_update_thread(self):
        while self.game_running:
            self.root.after(0, self.update_countdown)
            time.sleep(0.2)  # Update 5 times per second for smoother countdown
    
    def update_countdown(self):
        if not self.game_running:
            return
            
        now = datetime.now()
        
        # Calculate time remaining (read-only, no lock needed for these reads)
        try:
            if self.engine.game_start_time:
                total_seconds = self.cfg.ticks * self.cfg.tick_interval_seconds
                elapsed = (now - self.engine.game_start_time).total_seconds()
                remaining = max(0, total_seconds - elapsed)
                
                # Update progress bar and timer
                try:
                    tick_value = self.engine.tick  # Quick read
                    self.progress_bar['value'] = tick_value
                    self.time_label.config(text=f"{int(remaining)}s")
                except tk.TclError:
                    pass
        except (AttributeError, TypeError):
            pass  # Engine not fully initialized
        
        # Update side bet countdowns and immediately remove expired ones
        for sid, widgets in list(self.sidebet_widgets.items()):
            try:
                # Get side bet with lock
                with self.update_lock:
                    sb = self.engine.sidebets.get(sid)
                    if sb:
                        remaining = (sb.expiry_time - now).total_seconds()
                    else:
                        remaining = -1
                
                if remaining > 0:
                    widgets['countdown'].config(text=f"⏱ {int(remaining)}s")
                else:
                    # Immediately remove expired side bet
                    self.remove_sidebet_widget(sid)
            except (KeyError, tk.TclError, AttributeError):
                # Widget already destroyed or error, skip
                pass
    
    def update_display(self):
        if not self.game_running:
            return
            
        try:
            # Read data inside lock
            with self.update_lock:
                t = self.engine.tally
                
                # Detect changes for flash animation (with safety check)
                try:
                    changed_a_oranges = t.oranges_A > self.last_tally.oranges_A
                    changed_a_apples = t.lemons_A > self.last_tally.lemons_A
                    changed_b_oranges = t.oranges_B > self.last_tally.oranges_B
                    changed_b_apples = t.lemons_B > self.last_tally.lemons_B
                except (AttributeError, TypeError):
                    # last_tally not properly initialized
                    changed_a_oranges = changed_a_apples = False
                    changed_b_oranges = changed_b_apples = False
                
                # Get quotes
                main_px, side_prices = self.engine.current_quotes()
                self.last_main_px = main_px
                self.last_side_px = dict(side_prices)
                
                # Get current side bet IDs
                current_sids = set(self.engine.sidebets.keys())
            
            # Update UI OUTSIDE lock
            try:
                self.team_a_oranges.config(text=f"🍊 Oranges: {t.oranges_A}")
                self.team_a_apples.config(text=f"🍎 Apples: {t.lemons_A}")
                self.team_b_oranges.config(text=f"🍊 Oranges: {t.oranges_B}")
                self.team_b_apples.config(text=f"🍎 Apples: {t.lemons_B}")
                
                # Flash animation for changes
                if changed_a_oranges:
                    self.flash_widget(self.team_a_oranges)
                if changed_a_apples:
                    self.flash_widget(self.team_a_apples)
                if changed_b_oranges:
                    self.flash_widget(self.team_b_oranges)
                if changed_b_apples:
                    self.flash_widget(self.team_b_apples)
                
                # Update last tally
                self.last_tally = t.copy()
            except tk.TclError:
                return  # Widgets destroyed
            
            try:
                self.main_price_label.config(text=f"Price: {main_px:,}")
            except (tk.TclError, AttributeError):
                return
            
            # Update side bets (with protection)
            try:
                displayed_sids = set(self.sidebet_widgets.keys())
            except (AttributeError, RuntimeError):
                return
            
            # Remove expired (outside lock)
            for sid in list(displayed_sids - current_sids):
                try:
                    self.remove_sidebet_widget(sid)
                except Exception:
                    pass
            
            # Add new (outside lock)
            for sid in list(current_sids - displayed_sids):
                try:
                    self.add_sidebet_widget(sid)
                except Exception:
                    pass
            
            # Update prices
            for sid in list(current_sids):
                if sid in self.sidebet_widgets:
                    try:
                        px = side_prices.get(sid, 0)
                        if 'price' in self.sidebet_widgets[sid]:
                            self.sidebet_widgets[sid]['price'].config(text=f"Price: {px:,}")
                    except (KeyError, tk.TclError, RuntimeError, AttributeError):
                        pass  # Widget already removed
        except Exception as e:
            print(f"Error updating display: {e}")
    
    def add_sidebet_widget(self, sid: str):
        # Get side bet info with lock
        with self.update_lock:
            sb = self.engine.sidebets.get(sid)
            if not sb:
                return
            # Copy needed data
            description = sb.description
            price = sb.price
        
        # Create widgets outside lock
        # Alternate between left and right panels
        num_widgets = len(self.sidebet_widgets)
        if num_widgets % 2 == 0:
            container = self.left_sidebets_container
        else:
            container = self.right_sidebets_container
        
        frame = tk.Frame(container, bg='#f0f0f0', 
                        relief=tk.RAISED, bd=2)
        frame.pack(fill=tk.X, pady=3, padx=3)
        
        header = tk.Frame(frame, bg='#f0f0f0')
        header.pack(fill=tk.X, padx=5, pady=5)
        
        # Description only (no SB code)
        desc_label = tk.Label(header, text=description, 
                             font=('Arial', 12, 'bold'), 
                             bg='#f0f0f0', fg='#8800cc', 
                             wraplength=220, justify=tk.LEFT)
        desc_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        countdown = tk.Label(header, text="⏱ 10s", font=('Arial', 11, 'bold'), 
                           bg='#f0f0f0', fg='#0066cc')
        countdown.pack(side=tk.RIGHT)
        
        price_label = tk.Label(frame, text=f"Price: {price:,}", 
                              font=('Arial', 14, 'bold'), bg='#f0f0f0', fg='#000000')
        price_label.pack(pady=5)
        
        controls = tk.Frame(frame, bg='#f0f0f0')
        controls.pack(pady=5)
        
        tk.Label(controls, text="Qty:", font=('Arial', 11, 'bold'), 
                bg='#f0f0f0', fg='#000000').pack(side=tk.LEFT, padx=3)
        
        qty_entry = tk.Entry(controls, font=('Arial', 12, 'bold'), width=4,
                           justify='center')
        qty_entry.pack(side=tk.LEFT, padx=3)
        qty_entry.insert(0, "1")
        
        buy_btn = tk.Button(controls, text="BUY", font=('Arial', 11, 'bold'), 
                           bg='#00ff00', fg='black', width=6,
                           activebackground='#00cc00', relief=tk.RAISED, bd=2,
                           cursor='hand2',
                           command=lambda: self.execute_trade('buy', sid))
        buy_btn.pack(side=tk.LEFT, padx=3)
        
        sell_btn = tk.Button(controls, text="SELL", font=('Arial', 11, 'bold'), 
                            bg='#ff4400', fg='#000000', width=6,
                            activebackground='#cc3300', relief=tk.RAISED, bd=2,
                            cursor='hand2',
                            command=lambda: self.execute_trade('sell', sid))
        sell_btn.pack(side=tk.LEFT, padx=3)
        
        self.sidebet_widgets[sid] = {
            'frame': frame,
            'countdown': countdown,
            'price': price_label,
            'qty_entry': qty_entry
        }
    
    def remove_sidebet_widget(self, sid: str):
        if sid in self.sidebet_widgets:
            try:
                self.sidebet_widgets[sid]['frame'].destroy()
            except tk.TclError:
                pass  # Already destroyed
            del self.sidebet_widgets[sid]
    
    def flash_widget(self, widget):
        """Flash a widget briefly to show it changed."""
        original_bg = widget.cget('bg')
        try:
            # Flash to yellow
            widget.config(bg='#ffff00')
            # Return to original after 200ms
            self.root.after(200, lambda: self._restore_bg(widget, original_bg))
        except tk.TclError:
            pass
    
    def _restore_bg(self, widget, original_bg):
        """Restore widget background color."""
        try:
            widget.config(bg=original_bg)
        except tk.TclError:
            pass
    
    def show_status(self, message: str, status_type: str = 'success'):
        """Show status message in status bar (non-blocking)."""
        def update():
            try:
                if status_type == 'success':
                    self.status_bar.config(text=f"✓ {message}", 
                                         bg='#e8f5e8', fg='#006600')
                elif status_type == 'error':
                    self.status_bar.config(text=f"✗ {message}", 
                                         bg='#ffe8e8', fg='#cc0000')
                else:
                    self.status_bar.config(text=message, 
                                         bg='#e8e8e8', fg='#000000')
                # Clear after 3 seconds
                self.root.after(3000, lambda: self.status_bar.config(
                    text="Ready to trade", bg='#e8f5e8', fg='#006600'))
            except tk.TclError:
                pass  # Window closed
        
        self.root.after(0, update)
    
    def execute_trade(self, side: str, instrument: str):
        trade_msg = None
        error_msg = None
        
        try:
            if instrument == 'main':
                qty_str = self.main_qty_entry.get()
                qty = int(qty_str)
                if qty <= 0:
                    raise ValueError("Quantity must be positive")
                
                with self.update_lock:
                    price = self.last_main_px
                    if price == 0:
                        raise ValueError("No price available yet")
                    
                    mult = 1 if side == 'buy' else -1
                    self.engine.pos.main_qty += mult * qty
                    self.engine.pos.main_trades.append((mult * qty, price))
                    
                    # Track last trade
                    self.last_trade_info = {
                        'type': 'main',
                        'side': side,
                        'qty': qty,
                        'price': price
                    }
                    
                trade_msg = f"{side.upper()} MAIN x{qty} @ {price:,}"
            else:
                # Side bet trade - get qty and check widget BEFORE any lock
                if instrument not in self.sidebet_widgets:
                    error_msg = f"{instrument} widget no longer available!"
                    raise ValueError(error_msg)
                
                # Get quantity entry BEFORE acquiring lock
                try:
                    qty_str = self.sidebet_widgets[instrument]['qty_entry'].get()
                    qty = int(qty_str)
                    if qty <= 0:
                        raise ValueError("Quantity must be positive")
                except (KeyError, tk.TclError):
                    error_msg = f"{instrument} widget was removed!"
                    raise ValueError(error_msg)
                
                # Now acquire lock ONCE for all checks and trade execution
                with self.update_lock:
                    # Check if side bet still exists
                    if instrument not in self.engine.sidebets:
                        error_msg = f"{instrument} has expired!"
                        raise ValueError(error_msg)
                    
                    # Check expiry time
                    sb = self.engine.sidebets[instrument]
                    if datetime.now() >= sb.expiry_time:
                        error_msg = f"{instrument} just expired!"
                        raise ValueError(error_msg)
                    
                    # Get price
                    price = self.last_side_px.get(instrument, 0)
                    if price == 0:
                        raise ValueError("No price available yet")
                    
                    # Execute trade
                    mult = 1 if side == 'buy' else -1
                    current = self.engine.pos.side_qties.get(instrument, 0)
                    self.engine.pos.side_qties[instrument] = current + mult * qty
                    
                    if instrument not in self.engine.pos.side_trades:
                        self.engine.pos.side_trades[instrument] = []
                    self.engine.pos.side_trades[instrument].append((mult * qty, price))
                    
                    # Track last trade
                    self.last_trade_info = {
                        'type': 'sidebet',
                        'side': side,
                        'qty': qty,
                        'price': price,
                        'instrument': instrument,
                        'description': sb.description
                    }
                    
                    trade_msg = f"{side.upper()} {instrument} x{qty} @ {price:,}"
            
            # Update display using thread-safe method
            self.root.after(0, self.update_display)
            
            # Show success message in status bar
            if trade_msg:
                self.show_status(trade_msg, 'success')
                
        except ValueError as e:
            error_msg = str(e) if not error_msg else error_msg
            self.show_status(f"ERROR: {error_msg}", 'error')
        except Exception as e:
            self.show_status(f"ERROR: Trade failed - {str(e)}", 'error')
    
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
        
        # Simplified side bet bounds (more complex calculation possible)
        side_min = side_max = 0
        
        return int(pnl_main_min + side_min), int(pnl_main_max + side_max)
    
    def show_settlement(self):
        self.game_running = False
        
        t = self.engine.tally
        settle_main = self.engine.settle_value()
        
        dialog = tk.Toplevel(self.root)
        dialog.title("SETTLEMENT")
        dialog.geometry("600x500")
        dialog.configure(bg='#1e1e1e')
        
        tk.Label(dialog, text="🎯 FINAL SETTLEMENT 🎯", 
                font=('Arial', 18, 'bold'), bg='#1e1e1e', 
                fg='#00ff00').pack(pady=20)
        
        text_widget = scrolledtext.ScrolledText(dialog, font=('Courier', 11), 
                                               bg='#2d2d2d', fg='white', 
                                               width=60, height=20)
        text_widget.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
        
        text_widget.insert(tk.END, f"Final Totals:\n")
        text_widget.insert(tk.END, f"  Oranges: {t.oranges_total} " + 
                          f"(A:{t.oranges_A} / B:{t.oranges_B})\n")
        text_widget.insert(tk.END, f"  Apples:  {t.lemons_total} " + 
                          f"(A:{t.lemons_A} / B:{t.lemons_B})\n\n")
        text_widget.insert(tk.END, f"MAIN Settlement: {settle_main:,}\n")
        text_widget.insert(tk.END, "=" * 50 + "\n\n")
        
        realized = 0
        
        # MAIN P&L
        qty_main = sum(q for (q, _) in self.engine.pos.main_trades)
        if qty_main != 0:
            vwap_num = sum(q * p for (q, p) in self.engine.pos.main_trades)
            vwap = int(round(vwap_num / qty_main))
            pnl_main = qty_main * (settle_main - vwap)
            realized += pnl_main
            
            text_widget.insert(tk.END, f"MAIN P&L:\n")
            text_widget.insert(tk.END, f"  Qty: {qty_main}, VWAP: {vwap:,}\n")
            text_widget.insert(tk.END, f"  P&L: {pnl_main:,}\n\n")
        
        # Side bet P&L
        for sid, trades in self.engine.pos.side_trades.items():
            qty = sum(q for (q, _) in trades)
            if qty == 0:
                continue
            
            vwap_num = sum(q * p for (q, p) in trades)
            vwap = int(round(vwap_num / qty))
            
            sb = self.engine.get_sidebet(sid)
            if not sb:
                continue
            
            payoff = self.engine.side_payoff(sb)
            pnl = qty * (payoff - vwap)
            realized += pnl
            
            text_widget.insert(tk.END, f"{sid} [{sb.description}]:\n")
            text_widget.insert(tk.END, f"  Qty: {qty}, VWAP: {vwap:,}, " + 
                              f"Payoff: {payoff:,}\n")
            text_widget.insert(tk.END, f"  P&L: {pnl:,}\n\n")
        
        text_widget.insert(tk.END, "=" * 50 + "\n")
        text_widget.insert(tk.END, f"TOTAL P&L: {realized:,}\n")
        
        text_widget.config(state=tk.DISABLED)
        
        tk.Button(dialog, text="CLOSE", font=('Arial', 14), 
                 command=self.root.quit).pack(pady=20)


def main():
    root = tk.Tk()
    cfg = GameConfig()
    app = TradingGUI(root, cfg)
    root.mainloop()


if __name__ == "__main__":
    main()
