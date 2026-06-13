# Symulacja N-Ciałowa Układu Słonecznego

> **N-body gravitational simulation of the Solar System** — fully interactive,
> physically accurate, with switchable reference frames and adaptive numerical integration.
---

## Opis projektu

Program symuluje **9 ciał niebieskich** (Słońce + 8 planet) w płaszczyźnie ekliptyki,
modelując ich wzajemne grawitacyjne oddziaływania na podstawie
[Newtonowskiego prawa powszechnego ciążenia](https://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation).

Kluczowe właściwości:

- **Pełny n-ciałowy problem grawitacyjny** — wszystkie ciała wzajemnie na siebie
  oddziałują (nie tylko z~Słońcem)
-  **Rzeczywiste dane fizyczne** — masy, parametry orbitalne z katalogu NASA
-  **Wybór układu odniesienia** — dowolna planeta jako centrum widoku
  (geocentryczny, heliocentryczny, jowiocentryczny…)
-  **Metoda RK4** — 4. rząd dokładności, błąd energii < 0.02% w skali 200 lat
-  **Interaktywna wizualizacja** — pygame, ślady orbit, zoom itd.

---

## Instalacja

### Wymagania wstępne

- Python 3.8 lub nowszy
- pip

### Kroki

- Sklonuj repozytorium z githuba
- (Opcjonalnie) Utwórz wirtualne środowisko
- Zainstaluj wymagane pakiety z pliku requirements (numpy i pygame)
- Uruchom symulację

---

## Sterowanie

| Klawisz / Akcja   | Opis                                              |
|-------------------|---------------------------------------------------|
| `1` – `9`         | Wybór układu odniesienia (1 = Słońce, 4 = Ziemia…) |
| Scroll myszy      | Zoom in / out                                     |
| `G` / `H`         | Przyspieszenie / spowolnienie symulacji           |
| Strzałki / `WASD` | Przesuwanie kamery (pan)                          |
| `P`               | Pauza / wznowienie                                |
| `T`               | Ślady orbit — włącz/wyłącz                       |
| `N`               | Nazwy ciał — włącz/wyłącz                        |
| `O`               | Okręgi pomocnicze orbit — włącz/wyłącz           |
| `R`               | Reset widoku i śladów do wartości domyślnych     |
| `Q` / `Esc`       | Wyjście z programu                                |

### Wskazówka dotycząca układu odniesienia

Gdy wybierzesz **Ziemię (klawisz 4)** jako układ odniesienia, Ziemia
jest nieruchoma na środku ekranu, a orbity innych planet widoczne są
z perspektywy ziemskiego obserwatora — zobaczysz klasyczne **pętle retrograde**
Marsa i Jowisza, które przez wieki kłopotały astronomów.

---

## Podstawy fizyczne

### Prawo powszechnego ciążenia

Przyspieszenie *i*-tego ciała od wszystkich pozostałych:

```
a⃗ᵢ = G · Σⱼ≠ᵢ  mⱼ · (r⃗ⱼ − r⃗ᵢ) / (|r⃗ⱼ − r⃗ᵢ|² + ε²)^(3/2)
```

Parametr zmiękczający `ε = 1.5 × 10⁹ m ≈ 2R☉` zapobiega osobliwościom
numerycznym przy bliskich przelotach.

### Warunki początkowe

Każda planeta startuje w **peryhelium** swojej orbity kepleriańskiej.
Prędkość peryhelialna wyprowadzona z zasad zachowania energii i momentu pędu:

```
vₚ = √[ G·M☉·(1 + e) / (a·(1 − e)) ]
```

Zastosowano korektę środka masy (Σ mᵢ·vᵢ = 0), by układ nie dryfował.

### Weryfikacja — III prawo Keplera

Symulowane okresy obiegów zgodne planet są w przybliżeniu zgodne z tymi wyliczonymi analitycznie.

---

## Dlaczego algorytm RK4?

### Krótka odpowiedź

RK4 oferuje **optymalny kompromis** między dokładnością (rząd 4), prostotą
implementacji a kosztem obliczeniowym (4 ewaluacje sił na krok).

### Szczegółowe uzasadnienie

#### Alternatywy i ich wady

**Metoda Eulera (rząd 1)**
```
y_{n+1} = yₙ + h·f(yₙ)           błąd globalny: O(h)
```
- 1 ewaluacja sił na krok
- ❌ Energia rośnie monotonicznie — po kilku orbitach planeta „ucieka" ze słonecznego układu
- ❌ Niestabilna już przy h = 5 dób dla Merkurego

**Velocity Verlet / Leapfrog (rząd 2)**
```
r_{n+1} = rₙ + vₙh + ½aₙh²
v_{n+1} = vₙ + ½(aₙ + a_{n+1})h    błąd globalny: O(h²)
```
- 1 ewaluacja sił na krok
- ✅ **Symplektyczny** — zachowuje objętość przestrzeni fazowej, stąd brak dryfu energii
- ❌ Dwukrotnie mniej dokładny od RK4 przy tym samym kroku
- ✅ Lepszy niż RK4 dla symulacji > 10⁶ lat

**RK4 (rząd 4)** ← *zastosowany w projekcie*
```
y_{n+1} = yₙ + h/6·(k₁ + 2k₂ + 2k₃ + k₄)    błąd globalny: O(h⁴)
```
- 4 ewaluacje sił na krok
- ✅ Błąd zmniejsza się 16× przy 2× zmniejszeniu kroku
- ✅ Doskonała dokładność dla horyzontu lat–tysięcy lat
- ✅ Prosta, dobrze znana implementacja
- ⚠️ Nie jest symplektyczny → przy bardzo długich symulacjach (> 10⁶ lat) wolno traci energię

#### Dlaczego RK4 jest właściwy dla tego projektu

Metoda symplektyczna (Verlet) byłaby lepsza *tylko* w skali milionów lat.
>Dla interaktywnej wizualizacji układu słonecznego RK4 to wybór optymalny.

---

## Struktura projektu

```
solar-system-simulation/
│
├── main.py                  # Główny plik symulacji (pygame)
├── requirements.txt         # Zależności Python
├── .gitignore               # Pliki ignorowane przez Git
├── LICENSE                  # Licencja MIT
└── README.md                # Ten plik
```

### Główne sekcje kodu (`solar_system.py`)

| Sekcja | Linie    | Opis |
|--------|----------|------|
| 1 Stałe fizyczne | ~40–60   | G, AU, DAY, EPS², MAX_PHYS_DT |
| 2 Dane ciał | ~60–105  | Masy, kolory, parametry orbitalne |
| 3 Parametry okna i wizualizacji| ~105-120 |Parametry okna symulacji|
| 3 Warunki początkowe | ~125–190 | `initial_conditions()` |
| 4 Fizyka n-ciałowa | ~190–300 | `compute_accel()`, `rk4_step()`, `total_energy()` |
| 5 Funkcje rysowania | ~310–580 | gwiazdy, glow, pierścienie, ślady, HUD |
| 6 Pętla główna | ~580–820 | obsługa zdarzeń, substepping, render |

---

## Dane fizyczne (stałe)

| Stała | Wartość | Opis |
|-------|---------|------|
| G | 6.674 × 10⁻¹¹ m³ kg⁻¹ s⁻² | Stała grawitacyjna |
| AU | 1.496 × 10¹¹ m | Jednostka astronomiczna |
| M☉ | 1.989 × 10³⁰ kg | Masa Słońca |
| ε | 1.5 × 10⁹ m | Parametr zmiękczający |

Dane fizyczne planet: [NASA Planetary Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/).
Strona jest niestety czasowo niedostępna standardowo, ale można użyć [Internet Archive](https://archive.org/).


---

## Możliwe rozszerzenia

- [ ] Dodanie Księżyca Ziemi i księżyców Jowisza (Io, Europa, Ganimedes, Kallisto)
- [ ] Orbity komet (ekscentryczność bliska 1 lub > 1)
- [ ] Metoda symplektyczna (Forest-Ruth) dla długich symulacji
- [ ] Zapis / odczyt stanu symulacji (JSON)
- [ ] Symulacja 3D z OpenGL / Matplotlib 3D
- [ ] Efekty ogólnorelatywistyczne (precesja Merkurego ~43"/stulecie)

---

## Licencja

Projekt dostępny na licencji **MIT** — szczegóły w pliku [`LICENSE`](LICENSE).

---

## Podziękowania

- **Newton** (1687) — za prawo powszechnego ciążenia
- **Kepler** (1619) — za trzy prawa ruchu planet
- **Runge & Kutta** (1901) — za metodę całkowania 4. rzędu
- Twórcy `pygame` i `NumPy` — za niezastąpione biblioteki
