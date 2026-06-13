#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  SYMULACJA N-CIAŁOWA UKŁADU SŁONECZNEGO
================================================================================
  Opis:
    Program symuluje ruch 9 ciał niebieskich (Słońce + 8 planet) w płaszczyźnie
    ekliptyki na podstawie Newtonowskiego prawa powszechnego ciążenia.
    Całkowanie numeryczne wykonywane jest metodą Runge-Kutty 4. rzędu (RK4).
    Użytkownik może wybrać dowolne ciało jako punkt odniesienia układu —
    symulacja prowadzona jest zawsze w inercjalnym układzie środka masy,
    a transformacja do wybranego układu odniesienia wykonywana jest jedynie
    na potrzeby wizualizacji (bez pseudosił).

  Wymagania:
    Python >= 3.8
    pip install pygame numpy

  Sterowanie:
    1–9          Wybierz układ odniesienia (1=Słońce … 9=Neptun)
    Scroll myszy Zoom in / out
    G / H        Przyspieszenie / spowolnienie symulacji
    Strzałki     Przesuwanie kamery (pan)
    P            Pauza / wznowienie
    T            Ślady orbit — włącz/wyłącz
    N            Nazwy ciał — włącz/wyłącz
    O            Okręgi pomocnicze orbit — włącz/wyłącz
    R            Reset widoku i śladów
    Q / Esc      Wyjście z programu
================================================================================
"""

# ── Importy ───────────────────────────────────────────────────────────────────
import sys
import math
import random
from collections import deque

import numpy as np
import pygame

# ════════════════════════════════════════════════════════════════════════════════
# 1. STAŁE FIZYCZNE
# ════════════════════════════════════════════════════════════════════════════════

G    = 6.67430e-11        # Stała grawitacyjna Newtona        [m³ kg⁻¹ s⁻²]
AU   = 1.49597870700e11   # Jednostka astronomiczna           [m]
DAY  = 86_400.0           # Liczba sekund w dobie             [s]
YEAR = 365.25 * DAY       # Rok juliański                     [s]

# Parametr zmiękczający ε² zapobiega osobliwościom przy bliskich przelotach.
# Wartość ε ≈ 1,5 × 10⁹ m to ok. 2× promień Słońca — nie wpływa na normalne orbity.
EPS2 = (1.5e9) ** 2       # ε²  [m²]

# Maksymalny fizyczny krok całkowania [s] — gwarantuje stabilność RK4
# nawet przy największym mnożniku prędkości.
# Wartość 2.5 doby ≈ T_Merkurego / 35 ≈ bezpieczny margines.
# (Teoretyczne max dla RK4: ~39 dób, ale przy T/35 błąd orbity jest pomijalny)
MAX_PHYS_DT  = 2.5 * DAY    # [s]
MAX_SUBSTEPS = 300           # Górny limit podkroków na klatkę (cap wydajności)

# ════════════════════════════════════════════════════════════════════════════════
# 2. DANE FIZYCZNE CIAŁ NIEBIESKICH  (rzeczywiste wartości)
# ════════════════════════════════════════════════════════════════════════════════

# Krotka: (nazwa, masa [kg], promień_rys [px], kolor (R,G,B), ma_pierścień?)
BODY_DEF = [
    ("Słońce",   1.98892e30, 18, (255, 240,  50), False),   # M☉ = 1.989×10³⁰ kg
    ("Merkury",  3.30104e23,  4, (169, 169, 169), False),   # 0.0553 M⊕
    ("Wenus",    4.86732e24,  7, (255, 198,  73), False),   # 0.815  M⊕
    ("Ziemia",   5.97219e24,  7, ( 65, 135, 200), False),   # 1.000  M⊕ (masa referencyjna)
    ("Mars",     6.41693e23,  5, (210,  80,  40), False),   # 0.107  M⊕
    ("Jowisz",   1.89819e27, 14, (210, 165, 100), False),   # 317.8  M⊕
    ("Saturn",   5.68319e26, 11, (215, 195, 135),  True),   # 95.16  M⊕
    ("Uran",     8.68103e25,  8, (140, 210, 230), False),   # 14.54  M⊕
    ("Neptun",   1.02410e26,  8, ( 63,  90, 188), False),   # 17.15  M⊕
]

# Parametry orbitalne Keplera: (a [AU], e, φ₀ [rad])
#   a   — długość półosi wielkiej
#   e   — ekscentryczność (0 = okrąg, <1 = elipsa)
#   φ₀  — kąt startowy (rozmieszcza planety na orbitach dla czytelności)
ORBIT_PARAMS = [
    (0.00000, 0.000000, 0.000),  # Słońce      — w centrum układu odniesienia
    (0.38710, 0.205630, 0.000),  # Merkury
    (0.72333, 0.006770, 0.800),  # Wenus
    (1.00000, 0.016710, 1.800),  # Ziemia
    (1.52366, 0.093410, 3.200),  # Mars
    (5.20336, 0.048390, 0.500),  # Jowisz
    (9.53707, 0.054150, 2.100),  # Saturn
    (19.1913, 0.047170, 4.300),  # Uran
    (30.0690, 0.008590, 1.200),  # Neptun
]

# Wyliczone stałe (nie edytować bezpośrednio)
N        = len(BODY_DEF)                                   # liczba ciał
NAMES    = [b[0] for b in BODY_DEF]                       # lista nazw
MASSES   = np.array([b[1] for b in BODY_DEF], dtype=np.float64)  # masy [kg]
R_PX     = [b[2] for b in BODY_DEF]                       # promienie rysowania [px]
COLORS   = [b[3] for b in BODY_DEF]                       # kolory RGB
HAS_RING = [b[4] for b in BODY_DEF]                       # flaga pierścieni

# ════════════════════════════════════════════════════════════════════════════════
# 3. PARAMETRY OKNA I WIZUALIZACJI
# ════════════════════════════════════════════════════════════════════════════════

W, H          = 1280, 800      # Rozmiar okna [px]
FPS           = 60              # Docelowe klatki na sekundę

SCALE_DEF     = 8.5e-11        # Startowa skala [px/m] — widać cały układ
SCALE_MIN     = 1.5e-13        # Minimum powiększenia
SCALE_MAX     = 8.0e-9         # Maksimum powiększenia
ZOOM_FAC      = 1.15            # Mnożnik jednego kroku zoomu

DT_BASE       = DAY             # Bazowy krok całkowania = 1 doba [s]
SPEED_DEF     = 3.0             # Startowy mnożnik prędkości [dni/klatkę]
SPEED_MIN     = 0.10
SPEED_MAX     = 600.0
SPEED_MUL     = 1.35            # Mnożnik +/−

PAN_PX        = 14              # Pikseli przesunięcia kamery na klatkę (strzałki)
MAX_TRAIL     = 2400            # Maks. długość historii śladów (próbki)
TRAIL_EVERY   = 2               # Zapisuj ślad co N kroków symulacji
N_STARS       = 300             # Liczba gwiazd tła

COL_BG        = (  4,   5,  20) # Kolor tła
TRAIL_ALPHA   = 200             # Maks. alpha śladu (0-255)

# ════════════════════════════════════════════════════════════════════════════════
# 4. WARUNKI POCZĄTKOWE
# ════════════════════════════════════════════════════════════════════════════════

def initial_conditions():
    """
    Oblicza pozycje i prędkości startowe wszystkich ciał.

    Każda planeta jest umieszczana w peryhelium orbity Keplera,
    obróconej o kąt φ₀ względem osi +x.

    Odległość peryhelialna i prędkość peryhelialna (wyprowadzone
    z całek ruchu dwuciałowego):

        r_p = a·(1 − e)
        v_p = √[ G·M_☉·(1 + e) / r_p ]

    Kierunek prędkości jest prostopadły do promienia wodzącego
    (ruch w kierunku CCW), co zapewnia właściwy zwrot orbity.

    Na końcu stosujemy korektę środka masy:
        v_CM = Σ(mᵢ·vᵢ) / Σmᵢ  →  vel -= v_CM
    Dzięki temu całkowity pęd układu jest zerowy
    i centrum symulacji nie dryfuje.

    Zwraca
    ------
    pos : ndarray (N, 2)  — pozycje [m]
    vel : ndarray (N, 2)  — prędkości [m/s]
    """
    pos = np.zeros((N, 2), dtype=np.float64)
    vel = np.zeros((N, 2), dtype=np.float64)

    M_sun = MASSES[0]   # masa Słońca dominuje (99.86% masy układu)

    for i, (a_au, e, phi0) in enumerate(ORBIT_PARAMS):
        if i == 0:
            continue    # Słońce — pozostanie w centrum (przed korektą CM)

        a_m = a_au * AU                               # półoś wielka [m]
        r_p = a_m * (1.0 - e)                         # odl. peryhelialna [m]
        # Prędkość peryhelialna — zachowanie energii + momentu pędu
        v_p = math.sqrt(G * M_sun * (1.0 + e) / r_p) # [m/s]

        cos_p, sin_p = math.cos(phi0), math.sin(phi0)

        pos[i] = [ r_p * cos_p,   r_p * sin_p]       # pozycja w peryhelium
        vel[i] = [-v_p * sin_p,   v_p * cos_p]       # prędkość ⊥ do r̄

    # ── Korekta środka masy — zerowanie całkowitego pędu ──────────────────────
    total_mass = MASSES.sum()
    v_cm = np.einsum('i,ix->x', MASSES, vel) / total_mass  # prędkość CM [m/s]
    vel -= v_cm[np.newaxis, :]                              # teraz Σmᵢvᵢ = 0
    # ─────────────────────────────────────────────────────────────────────────

    return pos, vel


# ════════════════════════════════════════════════════════════════════════════════
# 5. FIZYKA — N-CIAŁOWY PROBLEM GRAWITACYJNY
# ════════════════════════════════════════════════════════════════════════════════

def compute_accel(pos: np.ndarray) -> np.ndarray:
    """
    Oblicza przyspieszenia wszystkich N ciał (wektoryzowane numpy).

    Prawo grawitacji Newtona z parametrem zmiękczającym ε:

        ā_i = G · Σ_{j≠i}  m_j · (r̄_j − r̄_i)
                            ─────────────────────────────────────
                            (|r̄_j − r̄_i|² + ε²)^(3/2)

    Implementacja nie używa jawnych pętli po parach —
    obliczenia są w pełni wektoryzowane przy użyciu broadcastingu
    numpy, co daje optymalną wydajność dla małej liczby ciał.

    Parametry
    ---------
    pos : ndarray (N, 2)  — aktualne pozycje [m]

    Zwraca
    ------
    acc : ndarray (N, 2)  — przyspieszenia [m/s²]
    """
    # dr[i, j] = pos[j] − pos[i]  →  kształt (N, N, 2)
    dr = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]

    # Kwadraty odległości ze zmiękczeniem → kształt (N, N)
    d2 = (dr * dr).sum(axis=-1) + EPS2

    # (|r|² + ε²)^{−3/2}, przekątna = 0 (brak samooddziaływania)
    inv3 = d2 ** (-1.5)
    np.fill_diagonal(inv3, 0.0)

    # acc[i, x] = G · Σ_j  m_j · inv3[i,j] · dr[i,j,x]
    acc = G * np.einsum('ij,j,ijx->ix', inv3, MASSES, dr)

    return acc


def rk4_step(pos: np.ndarray, vel: np.ndarray, dt: float):
    """
    Pojedynczy krok całkowania metodą Runge-Kutty 4. rzędu.

    Układ ODE:   dp/dt = v,   dv/dt = a(p)

    Cztery etapy (stadia):
        k₁ = f(t, y)
        k₂ = f(t + Δt/2, y + Δt/2 · k₁)
        k₃ = f(t + Δt/2, y + Δt/2 · k₂)
        k₄ = f(t + Δt,   y + Δt · k₃)

    Łączenie wyników:
        y_{n+1} = yₙ + Δt/6 · (k₁ + 2k₂ + 2k₃ + k₄)

    Globalny błąd: O(Δt⁴). Każdy krok wymaga 4 ewaluacji a(p).

    Parametry
    ---------
    pos : ndarray (N, 2)  — pozycje w chwili t
    vel : ndarray (N, 2)  — prędkości w chwili t
    dt  : float           — krok czasowy [s]

    Zwraca
    ------
    (new_pos, new_vel) : ndarray (N, 2), ndarray (N, 2)
    """
    # ── Stadium 1 ─────────────────────────────────────────────────────────────
    a1  = compute_accel(pos)
    k1p = vel
    k1v = a1

    # ── Stadium 2 ─────────────────────────────────────────────────────────────
    a2  = compute_accel(pos + 0.5 * dt * k1p)
    k2p = vel + 0.5 * dt * k1v
    k2v = a2

    # ── Stadium 3 ─────────────────────────────────────────────────────────────
    a3  = compute_accel(pos + 0.5 * dt * k2p)
    k3p = vel + 0.5 * dt * k2v
    k3v = a3

    # ── Stadium 4 ─────────────────────────────────────────────────────────────
    a4  = compute_accel(pos + dt * k3p)
    k4p = vel + dt * k3v
    k4v = a4

    # ── Złożenie wyniku ──────────────────────────────────────────────────────
    new_pos = pos + (dt / 6.0) * (k1p + 2.0*k2p + 2.0*k3p + k4p)
    new_vel = vel + (dt / 6.0) * (k1v + 2.0*k2v + 2.0*k3v + k4v)

    return new_pos, new_vel


def total_energy(pos: np.ndarray, vel: np.ndarray) -> float:
    """
    Całkowita energia mechaniczna układu.

        E = ½·Σᵢ mᵢ|vᵢ|²  −  G·Σ_{i<j} mᵢmⱼ/|rᵢ − rⱼ|
             (kinetyczna)         (potencjalna)

    Wartość ta powinna być niezmiennicza w czasie (zasada zachowania
    energii). Jej dryfowanie wskazuje na narastający błąd numeryczny.
    """
    # Energia kinetyczna
    Ek = 0.5 * float(np.einsum('i,ix->', MASSES, vel * vel))

    # Energia potencjalna (par: i < j)
    Ep = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r = float(np.linalg.norm(pos[j] - pos[i]))
            if r > 0:
                Ep -= G * MASSES[i] * MASSES[j] / r

    return Ek + Ep


# ════════════════════════════════════════════════════════════════════════════════
# 6. FUNKCJE POMOCNICZE RYSOWANIA
# ════════════════════════════════════════════════════════════════════════════════

def make_stars(n: int = N_STARS, seed: int = 2025) -> list:
    """Generuje listę pseudolosowych gwiazd tła (powtarzalne — stały seed)."""
    rng = random.Random(seed)
    stars = []
    for _ in range(n):
        x  = rng.randint(0, W - 1)
        y  = rng.randint(0, H - 1)
        br = rng.randint(70, 235)
        r  = rng.choices([1, 1, 1, 2], weights=[5, 5, 5, 1])[0]
        stars.append((x, y, br, r))
    return stars


def draw_stars(surf: pygame.Surface, stars: list) -> None:
    """Rysuje statyczne gwiazdy tła."""
    for x, y, br, r in stars:
        pygame.draw.circle(surf, (br, br, min(255, br + 20)), (x, y), r)


def w2s(wxy, ref_world, scale, cx=W//2, cy=H//2):
    """
    World-to-screen: zamienia fizyczne współrzędne [m] na piksele.

        sx = cx + (wx − rx) · scale
        sy = cy − (wy − ry) · scale    ← oś y odwrócona (ekran: y↓)

    Parametry
    ---------
    wxy       : array-like (2,)  — pozycja ciała [m]
    ref_world : array-like (2,)  — punkt odniesienia widoku [m]
    scale     : float            — skala [px/m]
    """
    rel = np.asarray(wxy) - np.asarray(ref_world)
    return (int(cx + rel[0] * scale),
            int(cy - rel[1] * scale))


def draw_glow(surf: pygame.Surface, cx: int, cy: int,
              color: tuple, r: int) -> None:
    """
    Renderuje efekt poświaty dla Słońca.

    Technika: kilka koncentrycznych okręgów z rosnącym promieniem
    i wykładniczo malejącą alfa, rysowanych na pomocowej powierzchni
    z kanałem alfa (SRCALPHA), a następnie nakładanych na ekran.
    """
    size = r * 10
    gsurf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
    ox, oy = size, size
    # Zewnętrzna poświata
    for step in range(8, 0, -1):
        gr  = int(r * (1.0 + step * 0.7))
        a   = max(4, int(48 // step))
        pygame.draw.circle(gsurf, (*color, a), (ox, oy), gr)
    # Jądro Słońca (jaśniejszy punkt w centrum)
    pygame.draw.circle(gsurf, (255, 255, 220, 180), (ox, oy), int(r * 0.6))
    surf.blit(gsurf, (cx - size, cy - size))


def draw_rings(surf: pygame.Surface, cx: int, cy: int, r: int) -> None:
    """
    Rysuje pierścienie Saturna jako dwie spłaszczone elipsy (perspektywa 2.5D).

    Wewnętrzny pierścień (jaśniejszy) i zewnętrzny (przezroczystszy).
    """
    for (factor, alpha) in [(2.5, 85), (1.85, 115)]:
        rx = int(r * factor)
        ry = max(3, int(r * 0.27))
        rsurf = pygame.Surface((rx * 2 + 8, ry * 2 + 8), pygame.SRCALPHA)
        # Gradient koloru pierścieni Saturna
        ring_col = (188, 165, 98, alpha)
        pygame.draw.ellipse(rsurf, ring_col, (0, 0, rx * 2, ry * 2), 3)
        surf.blit(rsurf, (cx - rx, cy - ry))


def draw_body(surf: pygame.Surface, cx: int, cy: int, idx: int) -> None:
    """
    Rysuje ciało niebieskie z odpowiednimi efektami wizualnymi:
    — Słońce: efekt glow
    — Saturn: pierścienie (elipsy)
    — Wszystkie: wypełniony okrąg + biała obwódka
    """
    r   = R_PX[idx]
    col = COLORS[idx]

    if idx == 0:         # Słońce — poświata
        draw_glow(surf, cx, cy, col, r)

    if HAS_RING[idx]:    # Saturn — pierścienie (tylna warstwa)
        draw_rings(surf, cx, cy, r)

    pygame.draw.circle(surf, col, (cx, cy), r)
    # Delikatna biała obwódka poprawia czytelność na ciemnym tle
    pygame.draw.circle(surf, (255, 255, 255), (cx, cy), r, 1)


def draw_trail(surf: pygame.Surface, history: deque,
               body_idx: int, ref_idx: int,
               scale: float, cam_offset: np.ndarray) -> None:
    """
    Rysuje ślad orbity ciała w wybranym układzie odniesienia.

    Dla każdego momentu k z historii oblicza współrzędne względne:

        rel_k = pos_k[body_idx] − pos_k[ref_idx] − cam_offset

    Dzięki temu ślad poprawnie odzwierciedla ruch relative do
    wybranego ciała odniesienia (nie do układu inercjalnego).

    Ślad blednie stopniowo: starsze punkty są ciemniejsze.
    """
    n = len(history)
    if n < 2:
        return

    base_col = COLORS[body_idx]
    pts      = []

    # Oblicz wszystkie punkty śladu w ekranowych współrzędnych
    for k, snap in enumerate(history):
        rel = snap[body_idx] - snap[ref_idx] - cam_offset
        sx  = int(W // 2 + rel[0] * scale)
        sy  = int(H // 2 - rel[1] * scale)
        # age: 0.0 = najstarszy, 1.0 = najnowszy
        age = (k + 1) / n
        pts.append((sx, sy, age))

    # Rysuj segmenty z narastającą jasnością (starsze = ciemniejsze)
    for k in range(1, len(pts)):
        age = pts[k][2]
        intensity = max(0.05, age ** 1.4)   # nieliniowe zanikanie
        r_c = max(8, int(base_col[0] * intensity))
        g_c = max(8, int(base_col[1] * intensity))
        b_c = max(8, int(base_col[2] * intensity))
        p0  = (pts[k-1][0], pts[k-1][1])
        p1  = (pts[k  ][0], pts[k  ][1])
        # Sprawdź czy punkt jest w obrębie ekranu (z marginesem)
        if (-500 < p0[0] < W+500 and -500 < p0[1] < H+500 and
                -500 < p1[0] < W+500 and -500 < p1[1] < H+500):
            pygame.draw.line(surf, (r_c, g_c, b_c), p0, p1, 1)


def draw_scale_bar(surf: pygame.Surface, fonts: dict, scale: float) -> None:
    """
    Rysuje linijkę skali (w AU) w lewym dolnym rogu ekranu.

    Wybiera automatycznie „ładną" wartość AU, której długość w pikselach
    mieści się w przedziale [60, 200] px, co zapewnia czytelność.
    """
    au_candidates = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]
    target_px = 120
    best_au   = 1.0
    best_diff = float('inf')
    for au in au_candidates:
        px   = au * AU * scale
        diff = abs(px - target_px)
        if diff < best_diff:
            best_diff = diff
            best_au   = au

    px_len = int(best_au * AU * scale)
    if 30 <= px_len <= 350:
        x1, y0 = 12, H - 25
        x2     = x1 + px_len
        col    = (180, 180, 200)
        # Pozioma linia z pionowymi zakończeniami
        pygame.draw.line(surf, col, (x1, y0), (x2, y0), 2)
        pygame.draw.line(surf, col, (x1, y0-5), (x1, y0+5), 2)
        pygame.draw.line(surf, col, (x2, y0-5), (x2, y0+5), 2)
        # Etykieta
        label = f"{best_au:.2g} AU".replace(".0 ", " ")
        lbl   = fonts['xs'].render(label, True, col)
        surf.blit(lbl, ((x1 + x2) // 2 - lbl.get_width() // 2, y0 - 18))


def fmt_time(seconds: float) -> str:
    """Formatuje czas symulacji [s] → czytelny ciąg znaków."""
    years = int(seconds / YEAR)
    days  = int((seconds % YEAR) / DAY)
    if years >= 1000:
        return f"{years:,} lat"
    if years > 0:
        return f"{years} lat {days} dni"
    return f"{days} dni"


def draw_ui(surf: pygame.Surface, fonts: dict,
            ref_idx: int, sim_time: float, speed: float,
            paused: bool, trails_on: bool, names_on: bool,
            orbits_on: bool, energy: float, energy0: float) -> None:
    """
    Rysuje nakładkę informacyjną (HUD) z czterema obszarami:

    • Lewy górny:   tytuł, układ odniesienia, czas symulacji
    • Prawy górny:  prędkość, energia, statusy trybów
    • Lewy dolny:   linijka skali (rysowana osobno)
    • Prawy dolny:  lista ciał z klawiszami wyboru
    """
    fsm = fonts['sm']
    fxs = fonts['xs']

    # ── Lewy górny ────────────────────────────────────────────────────────────
    y = 10
    surf.blit(fsm.render("SYMULACJA UKŁADU SŁONECZNEGO", True,
                         (170, 175, 225)), (12, y)); y += 24
    surf.blit(fsm.render(f"Układ odniesienia:  {NAMES[ref_idx]}",
                         True, COLORS[ref_idx]), (12, y)); y += 22
    surf.blit(fsm.render(f"Czas symulacji:  {fmt_time(sim_time)}",
                         True, (140, 200, 150)), (12, y)); y += 22
    if paused:
        surf.blit(fsm.render("⏸  PAUZA", True, (255, 215, 50)), (12, y))

    # ── Prawy górny ───────────────────────────────────────────────────────────
    if speed >= 10:
        speed_str = f"Prędkość: ×{speed:.0f} dni/kl"
    else:
        speed_str = f"Prędkość: ×{speed:.2g} dni/kl"

    n_sub_cur = min(MAX_SUBSTEPS, max(1, math.ceil(speed * DAY / MAX_PHYS_DT)))
    if n_sub_cur > 1:
        speed_str += f"  [{n_sub_cur} sub]"

    rel_err = abs((energy - energy0) / energy0) * 100 if energy0 != 0 else 0.0

    right_items = [
        (speed_str,                              (200, 200,  90)),
        (f"Energia: {energy:.3e} J",             ( 80, 185, 185)),
        (f"Błąd E:  {rel_err:.4f}%",             (100, 210, 120) if rel_err < 0.01
                                                  else (255, 150,  50)),
        (f"Ślady: {'■ ON' if trails_on else '□ OFF'}",
                                                 (100, 210, 100) if trails_on
                                                  else ( 90,  90, 110)),
        (f"Nazwy: {'■ ON' if names_on  else '□ OFF'}",
                                                 (100, 180, 220) if names_on
                                                  else ( 90,  90, 110)),
        (f"Orbity: {'■ ON' if orbits_on else '□ OFF'}",
                                                 (210, 140,  90) if orbits_on
                                                  else ( 90,  90, 110)),
    ]
    for i, (txt, col) in enumerate(right_items):
        tw = fsm.size(txt)[0]
        surf.blit(fsm.render(txt, True, col), (W - tw - 12, 10 + i * 22))

    # ── Dolny lewy: skróty klawiszowe ─────────────────────────────────────────
    help_keys = [
        ("1–9",       "Układ odniesienia"),
        ("Scroll",    "Zoom"),
        ("G  /  H",   "Prędkość"),
        ("Strzałki",  "Przesuń widok"),
        ("P",         "Pauza"),
        ("T / N / O", "Ślady / Nazwy / Orbity"),
        ("R",         "Reset widoku"),
        ("Q / Esc",   "Wyjście"),
    ]
    y0 = H - 24 - len(help_keys) * 17
    for i, (k, v) in enumerate(help_keys):
        ky = y0 + i * 17
        surf.blit(fxs.render(k, True, (220, 185,  80)), (12, ky))
        surf.blit(fxs.render(v, True, (135, 135, 160)), (80, ky))

    # ── Dolny prawy: lista ciał z klawiszami ──────────────────────────────────
    y2 = H - 10 - N * 17
    for i in range(N):
        active = (i == ref_idx)
        col    = COLORS[i] if active else (78, 78, 100)
        marker = "►" if active else f" {i+1}"
        txt    = f"{marker}  {NAMES[i]}"
        tw     = fxs.size(txt)[0]
        surf.blit(fxs.render(txt, True, col), (W - tw - 12, y2 + i * 17))


# ════════════════════════════════════════════════════════════════════════════════
# 7. GŁÓWNA PĘTLA PROGRAMU
# ════════════════════════════════════════════════════════════════════════════════

def main():
    """
    Punkt wejścia symulacji.

    Odpowiada za:
    1. Inicjalizację pygame i okna
    2. Obliczenie warunków początkowych
    3. Obsługę zdarzeń (klawiatura, mysz)
    4. Pętlę symulacji (krok RK4)
    5. Rysowanie klatki (tło → ślady → ciała → HUD)
    """
    # ── Inicjalizacja pygame ──────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(
        "Symulacja N-Ciałowa Układu Słonecznego  ·  RK4  ·  klawisz 1-9: układ odniesienia")
    clock = pygame.time.Clock()

    # ── Czcionki ──────────────────────────────────────────────────────────────
    try:
        fonts = {
            'sm': pygame.font.SysFont("consolas", 14),
            'xs': pygame.font.SysFont("consolas", 12),
        }
    except Exception:
        fonts = {
            'sm': pygame.font.Font(None, 20),
            'xs': pygame.font.Font(None, 17),
        }

    # ── Statyczne elementy tła ────────────────────────────────────────────────
    stars = make_stars()

    # ── Warunki początkowe ────────────────────────────────────────────────────
    pos, vel   = initial_conditions()
    energy0    = total_energy(pos, vel)   # referencyjna energia startowa
    energy     = energy0

    # ── Stan symulacji ────────────────────────────────────────────────────────
    sim_time   = 0.0
    speed      = SPEED_DEF      # [dni/klatkę]
    paused     = False

    # ── Stan widoku ───────────────────────────────────────────────────────────
    ref_idx    = 0               # 0 = Słońce jako domyślny układ odniesienia
    scale      = SCALE_DEF       # [px/m]
    cam_offset = np.zeros(2)     # dodatkowe przesunięcie kamery [m]

    # ── Tryby wyświetlania ────────────────────────────────────────────────────
    trails_on  = True
    names_on   = True
    orbits_on  = True

    # ── Historia śladów orbit ─────────────────────────────────────────────────
    # Deque przechowuje kopie tablicy pos (N, 2) dla każdej próbki
    trail_hist  = deque(maxlen=MAX_TRAIL)
    trail_cnt   = 0
    trail_hist.append(pos.copy())

    # ── Liczniki pomocnicze ───────────────────────────────────────────────────
    energy_cnt  = 0     # co ile klatek przeliczać energię (operacja kosztowna)

    # ══════════════════════════════════════════════════════════════════════════
    # PĘTLA GŁÓWNA
    # ══════════════════════════════════════════════════════════════════════════
    running = True
    while running:

        # ── Obsługa zdarzeń ───────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.KEYDOWN:
                k = ev.key

                # Wybór układu odniesienia: klawisze 1–9 (rząd + numpad)
                ref_map = {
                    pygame.K_1: 0, pygame.K_KP1: 0,
                    pygame.K_2: 1, pygame.K_KP2: 1,
                    pygame.K_3: 2, pygame.K_KP3: 2,
                    pygame.K_4: 3, pygame.K_KP4: 3,
                    pygame.K_5: 4, pygame.K_KP5: 4,
                    pygame.K_6: 5, pygame.K_KP6: 5,
                    pygame.K_7: 6, pygame.K_KP7: 6,
                    pygame.K_8: 7, pygame.K_KP8: 7,
                    pygame.K_9: 8, pygame.K_KP9: 8,
                }
                if k in ref_map:
                    ref_idx    = ref_map[k]
                    cam_offset = np.zeros(2)   # wyśrodkuj na nowe ciało odniesienia

                # Zoom klawiaturą
                elif k in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                    scale = min(SCALE_MAX, scale * ZOOM_FAC)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    scale = max(SCALE_MIN, scale / ZOOM_FAC)

                # Prędkość symulacji
                elif k == pygame.K_h:
                    speed = min(SPEED_MAX, speed * SPEED_MUL)
                elif k == pygame.K_g:
                    speed = max(SPEED_MIN, speed / SPEED_MUL)

                # Przełączniki trybów
                elif k == pygame.K_p:
                    paused = not paused
                elif k == pygame.K_t:
                    trails_on = not trails_on
                    if not trails_on:
                        trail_hist.clear()        # wyczyść ślady po wyłączeniu
                elif k == pygame.K_n:
                    names_on = not names_on
                elif k == pygame.K_o:
                    orbits_on = not orbits_on

                # Reset widoku i śladów
                elif k == pygame.K_r:
                    scale      = SCALE_DEF
                    cam_offset = np.zeros(2)
                    trail_hist.clear()
                    trail_hist.append(pos.copy())

                # Wyjście
                elif k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

            # Zoom kółkiem myszy (wyśrodkowany na kursor)
            elif ev.type == pygame.MOUSEWHEEL:
                if ev.y > 0:
                    scale = min(SCALE_MAX, scale * ZOOM_FAC)
                elif ev.y < 0:
                    scale = max(SCALE_MIN, scale / ZOOM_FAC)

        # ── Przesuwanie kamery strzałkami / WASD ─────────────────────────────
        kh      = pygame.key.get_pressed()
        pan_m   = PAN_PX / scale    # piksele → metry (zależne od skali)
        if kh[pygame.K_LEFT]  or kh[pygame.K_a]: cam_offset[0] -= pan_m
        if kh[pygame.K_RIGHT] or kh[pygame.K_d]: cam_offset[0] += pan_m
        if kh[pygame.K_DOWN]  or kh[pygame.K_s]: cam_offset[1] -= pan_m
        if kh[pygame.K_UP]    or kh[pygame.K_w]: cam_offset[1] += pan_m

        # ── Krok całkowania RK4 z adaptacyjnymi podkrokami ───────────────────
        # PROBLEM: przy dużym speed (np. ×200 dób/klatkę) dt_total = 200 dób
        # jest za duże dla RK4 — Merkury odczuwa silne przyspieszenie
        # w peryhelium i może być wyrzucony z orbity (niestabilność numeryczna).
        #
        # ROZWIĄZANIE — adaptive substepping:
        # Dzielimy dt_total na n_sub równych podkroków, z których każdy
        # ma co najwyżej MAX_PHYS_DT = 2,5 doby. Gwarantuje to stabilność
        # RK4 dla wszystkich prędkości symulatora.
        #
        # Np. speed = 600 → dt_total = 600 dób → n_sub = 240 × 2,5 dób
        if not paused:
            dt_total = DT_BASE * speed          # żądany krok klatki [s]
            n_sub    = min(MAX_SUBSTEPS,
                           max(1, math.ceil(dt_total / MAX_PHYS_DT)))
            dt_sub   = dt_total / n_sub         # krok każdego podkroku [s]
            for _ in range(n_sub):
                pos, vel = rk4_step(pos, vel, dt_sub)
            sim_time += dt_total

            # Zapis próbki do historii śladów
            trail_cnt += 1
            if trail_cnt >= TRAIL_EVERY:
                trail_hist.append(pos.copy())
                trail_cnt = 0

            # Energia — kosztowna operacja, obliczana co 90 klatek
            energy_cnt += 1
            if energy_cnt >= 90:
                energy     = total_energy(pos, vel)
                energy_cnt = 0

        # ── Wyznaczenie środka widoku ────────────────────────────────────────
        # ref_world: pozycja w metrach, która będzie wyśrodkowana na ekranie
        ref_world = pos[ref_idx] + cam_offset

        # ════════════════════════════════════════════════════════════════════
        # RENDEROWANIE KLATKI
        # ════════════════════════════════════════════════════════════════════

        # 1. Tło i gwiazdy
        screen.fill(COL_BG)
        draw_stars(screen, stars)

        # 2. Pomocnicze okręgi orbit (aproksymacja półosią wielką)
        if orbits_on:
            sun_sx, sun_sy = w2s(pos[0], ref_world, scale)  # pozycja Słońca na ekranie
            for i in range(1, N):
                a_au, _e, _phi = ORBIT_PARAMS[i]
                r_orb_px = int(a_au * AU * scale)
                if 3 <= r_orb_px <= max(W, H) * 4:
                    # Bardzo ciemna wersja koloru planety
                    oc = tuple(max(12, int(c * 0.18)) for c in COLORS[i])
                    pygame.draw.circle(screen, oc,
                                       (sun_sx, sun_sy), r_orb_px, 1)

        # 3. Ślady orbit
        if trails_on and len(trail_hist) >= 2:
            for i in range(N):
                if i == ref_idx:
                    continue    # ciało odniesienia zawsze w centrum — ślad byłby punktem
                draw_trail(screen, trail_hist, i, ref_idx, scale, cam_offset)

        # 4. Ciała niebieskie i etykiety
        for i in range(N):
            sx, sy = w2s(pos[i], ref_world, scale)
            # Rysuj tylko ciała w pobliżu widocznego obszaru
            if -400 < sx < W + 400 and -400 < sy < H + 400:
                draw_body(screen, sx, sy, i)
                if names_on:
                    lbl = fonts['xs'].render(NAMES[i], True, COLORS[i])
                    screen.blit(lbl, (sx + R_PX[i] + 4, sy - 7))

        # 5. Nakładka informacyjna (HUD)
        draw_scale_bar(screen, fonts, scale)
        draw_ui(screen, fonts, ref_idx, sim_time, speed, paused,
                trails_on, names_on, orbits_on, energy, energy0)

        # ── Wyświetlenie klatki ───────────────────────────────────────────────
        pygame.display.flip()
        clock.tick(FPS)

    # ── Sprzątanie ────────────────────────────────────────────────────────────
    pygame.quit()
    sys.exit(0)


# ── Punkt wejścia ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
