# NLF(1)-0416 — эталон для модели перехода SA-AFT

- `nlf0416_coord.dat` — координаты из NASA TP-1861 (Somers, 1981), таблица I
  (https://ntrs.nasa.gov/citations/19810015487).
- Эксперимент (Langley LTPT, M 0,10), снято с графиков TP-1861 (рис. 10, 12), точность ±0,02 x/c:
  при Re 4·10⁶, α = 0° (c_l ≈ 0,45) переход сверху x/c ≈ 0,43, снизу ≈ 0,60–0,62;
  при Re 4·10⁶ ламинарного отрывного пузыря на верхней поверхности нет.
- Расчёты NASA этой же моделью (FUN3D, OVERFLOW): Hildebrand et al., «Implementation and
  Verification of the SST-γ and SA-AFT Transition Models in FUN3D», AIAA AVIATION 2023
  (https://ntrs.nasa.gov/citations/20230004109): M 0,1, Re 4·10⁶, Tu 0,15 % (α 5°), ν_t/ν = 1.
