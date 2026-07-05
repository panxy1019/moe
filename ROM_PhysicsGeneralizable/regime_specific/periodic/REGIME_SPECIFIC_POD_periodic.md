# Regime-specific Area-weighted POD: periodic

## Regime Mapping

- Target regime: `periodic`
- Source labels: `['developing_periodic_shedding', 'mature_periodic_shedding', 'high_re_2d_periodic_near_modeA']`
- Number of Re cases: `63`
- Total snapshots: `10150`
- Re range: `60.3077454666` to `200`

## POD Method

- Per-Re mean subtraction is used before POD.
- Inner product uses lumped `point_areas` from `mesh_l2_point_area_weights.npz`.
- Randomized block SVD is applied directly to weighted raw snapshots; no global ROM tensors are reused.

## Outputs

- `phi_uv.shape = (80, 194736)`
- `coeff_uv.shape = (10150, 80)`
- `velocity captured energy = 9.999720490877e-01`
- `phi_p.shape = (80, 97368)`
- `coeff_p.shape = (10150, 80)`
- `pressure captured energy = 9.999819190345e-01`
- elapsed: `613.3 s`

## Cases

- `Re_60p307745`: Re=60.3077454666, source=`developing_periodic_shedding`, snapshots=162
- `Re_61p755954`: Re=61.7559536282, source=`developing_periodic_shedding`, snapshots=161
- `Re_63p499817`: Re=63.499816815, source=`developing_periodic_shedding`, snapshots=161
- `Re_65p259829`: Re=65.2598292316, source=`developing_periodic_shedding`, snapshots=161
- `Re_66p970112`: Re=66.9701121204, source=`developing_periodic_shedding`, snapshots=161
- `Re_68p649714`: Re=68.6497139288, source=`developing_periodic_shedding`, snapshots=161
- `Re_70p314635`: Re=70.3146353337, source=`developing_periodic_shedding`, snapshots=161
- `Re_71p972931`: Re=71.9729308789, source=`developing_periodic_shedding`, snapshots=161
- `Re_73p628287`: Re=73.6282869994, source=`developing_periodic_shedding`, snapshots=161
- `Re_75p282340`: Re=75.2823402819, source=`developing_periodic_shedding`, snapshots=161
- `Re_76p935803`: Re=76.9358029334, source=`developing_periodic_shedding`, snapshots=161
- `Re_78p588971`: Re=78.5889708164, source=`developing_periodic_shedding`, snapshots=161
- `Re_80p241943`: Re=80.2419430177, source=`developing_periodic_shedding`, snapshots=161
- `Re_81p894708`: Re=81.8947080155, source=`developing_periodic_shedding`, snapshots=161
- `Re_83p547164`: Re=83.5471642516, source=`developing_periodic_shedding`, snapshots=161
- `Re_85p199103`: Re=85.1991031321, source=`developing_periodic_shedding`, snapshots=161
- `Re_86p850168`: Re=86.8501684903, source=`developing_periodic_shedding`, snapshots=161
- `Re_88p499815`: Re=88.4998153335, source=`developing_periodic_shedding`, snapshots=161
- `Re_90p147341`: Re=90.1473406471, source=`developing_periodic_shedding`, snapshots=161
- `Re_91p792204`: Re=91.792204085, source=`developing_periodic_shedding`, snapshots=161
- `Re_93p435204`: Re=93.4352043333, source=`developing_periodic_shedding`, snapshots=161
- `Re_95p081752`: Re=95.0817519005, source=`developing_periodic_shedding`, snapshots=161
- `Re_96p749308`: Re=96.749307569, source=`developing_periodic_shedding`, snapshots=161
- `Re_98p480345`: Re=98.4803451202, source=`developing_periodic_shedding`, snapshots=161
- `Re_100p352251`: Re=100.352251335, source=`mature_periodic_shedding`, snapshots=161
- `Re_102p440042`: Re=102.440041809, source=`mature_periodic_shedding`, snapshots=161
- `Re_104p710911`: Re=104.710910987, source=`mature_periodic_shedding`, snapshots=161
- `Re_107p050737`: Re=107.050736557, source=`mature_periodic_shedding`, snapshots=161
- `Re_109p395985`: Re=109.395985153, source=`mature_periodic_shedding`, snapshots=161
- `Re_111p734011`: Re=111.734011486, source=`mature_periodic_shedding`, snapshots=161
- `Re_114p066308`: Re=114.066307867, source=`mature_periodic_shedding`, snapshots=161
- `Re_116p395488`: Re=116.395488165, source=`mature_periodic_shedding`, snapshots=161
- `Re_118p723173`: Re=118.723173369, source=`mature_periodic_shedding`, snapshots=161
- `Re_121p050171`: Re=121.05017082, source=`mature_periodic_shedding`, snapshots=161
- `Re_123p376833`: Re=123.376832843, source=`mature_periodic_shedding`, snapshots=161
- `Re_125p703274`: Re=125.703273692, source=`mature_periodic_shedding`, snapshots=162
- `Re_128p029461`: Re=128.029461168, source=`mature_periodic_shedding`, snapshots=161
- `Re_130p355225`: Re=130.355224828, source=`mature_periodic_shedding`, snapshots=161
- `Re_132p680203`: Re=132.680202796, source=`mature_periodic_shedding`, snapshots=161
- `Re_135p003744`: Re=135.003743604, source=`mature_periodic_shedding`, snapshots=161
- `Re_137p324830`: Re=137.324829556, source=`mature_periodic_shedding`, snapshots=162
- `Re_139p642302`: Re=139.642302099, source=`mature_periodic_shedding`, snapshots=162
- `Re_141p956319`: Re=141.956318757, source=`mature_periodic_shedding`, snapshots=161
- `Re_144p273459`: Re=144.273459297, source=`mature_periodic_shedding`, snapshots=161
- `Re_146p619578`: Re=146.619578296, source=`mature_periodic_shedding`, snapshots=161
- `Re_149p059229`: Re=149.059229449, source=`mature_periodic_shedding`, snapshots=161
- `Re_151p686208`: Re=151.686208001, source=`high_re_2d_periodic_near_modeA`, snapshots=161
- `Re_154p520852`: Re=154.520851959, source=`high_re_2d_periodic_near_modeA`, snapshots=161
- `Re_157p459588`: Re=157.45958766, source=`high_re_2d_periodic_near_modeA`, snapshots=161
- `Re_160p415176`: Re=160.415175616, source=`high_re_2d_periodic_near_modeA`, snapshots=161
- `Re_163p364123`: Re=163.364122702, source=`high_re_2d_periodic_near_modeA`, snapshots=161
- `Re_166p306373`: Re=166.306372744, source=`high_re_2d_periodic_near_modeA`, snapshots=161
- `Re_169p244893`: Re=169.244893107, source=`high_re_2d_periodic_near_modeA`, snapshots=161
- `Re_172p181708`: Re=172.181708206, source=`high_re_2d_periodic_near_modeA`, snapshots=161
- `Re_175p117940`: Re=175.117940142, source=`high_re_2d_periodic_near_modeA`, snapshots=161
- `Re_178p054368`: Re=178.05436806, source=`high_re_2d_periodic_near_modeA`, snapshots=161
- `Re_180p992055`: Re=180.992054605, source=`high_re_2d_periodic_near_modeA`, snapshots=161
- `Re_183p933395`: Re=183.933394636, source=`high_re_2d_periodic_near_modeA`, snapshots=161
- `Re_186p884600`: Re=186.884600344, source=`high_re_2d_periodic_near_modeA`, snapshots=162
- `Re_189p862278`: Re=189.86227838, source=`high_re_2d_periodic_near_modeA`, snapshots=161
- `Re_192p911664`: Re=192.911663952, source=`high_re_2d_periodic_near_modeA`, snapshots=162
- `Re_196p160723`: Re=196.160723205, source=`high_re_2d_periodic_near_modeA`, snapshots=161
- `Re_200p000000`: Re=200, source=`high_re_2d_periodic_near_modeA`, snapshots=162
