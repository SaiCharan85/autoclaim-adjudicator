# Assets

Where every image, icon and animation in the UI comes from. One brand palette throughout:
navy `#1e3a8a`, blue `#2563eb`, sky `#93c5fd`, mist `#eef2ff`, mint `#10b981`, amber `#f59e0b`,
coral `#ef4444`, ink `#1f2937`, paper `#f6f7fb` (`autoclaim.ui.assets.PALETTE`).

## Self-made (committed; no third-party rights)

| File | What | Used on |
|---|---|---|
| `assets/brand/logo.svg` | Shield with a car front and a verified tick, plus the `autoclaim` wordmark | Sidebar logo on both pages |
| `assets/brand/favicon.svg`, `favicon.png` | Shield with a tick; the PNG is rendered from the SVG | Browser tab icon, collapsed sidebar |
| `assets/brand/empty_queue.svg` | Empty tray with finished, ticked documents | Adjuster console when nothing waits for review |
| `assets/brand/onboarding.svg` | Car and its story -> policy check -> decision shield | Try a claim, before the first run |
| `assets/brand/error.svg` | Stalled car with a warning triangle and a cone | Try a claim, if a run fails |
| Loader, success tick, review-needed alert | Hand-written SVG + CSS animations (`autoclaim.ui.assets`); still under `prefers-reduced-motion` | Try a claim (while deciding; in the verdict card) |

## Icons: Lucide (committed)

- **Source:** the official Lucide GitHub release `lucide-icons-1.53.0.zip`
  (https://github.com/lucide-icons/lucide/releases/tag/1.53.0), copied by `scripts/fetch_icons.py`.
- **License:** ISC (`assets/icons/lucide/LICENSE`, copied from the same tag).
- **Icons (24):** `banknote`, `book-open`, `calculator`, `calendar`, `camera`, `car`, `circle-check`, `circle-x`, `clipboard-list`, `file-text`, `image`, `inbox`, `info`, `loader-circle`, `scale`, `search`, `shield-alert`, `shield-check`, `shuffle`, `signpost`, `sparkles`, `triangle-alert`, `user`, `user-check`.
- Every icon in the UI is Lucide, inlined as SVG so it takes the theme colour.

## Demo car-damage photos: Unsplash API (git-ignored)

- **Source:** the official Unsplash API (`/search/photos`, query "car accident damage"), key
  `UNSPLASH_ACCESS_KEY` in `.env`; `uv run python scripts/fetch_demo_images.py` regenerates them.
- **Terms followed:** images are hotlinked from the URLs the API returns (Unsplash's guidelines
  require it), each photo's download endpoint is triggered, and every photo is credited
  "Photo by <name> on Unsplash" with both links carrying `utm_source` / `utm_medium=referral`.
  Only the links and credits are stored (`assets/demo_claims/credits.json`, git-ignored).
- **Alternative:** `--provider pexels` (`PEXELS_API_KEY`) downloads the files into the folder
  instead and credits "Photo by <name> on Pexels".
- The UI labels them "Demo photo, not from this claim".

Current set (fetched 2026-10-08):

| # | Photographer | Photo | Description |
|---|---|---|---|
| 1 | [Anthony Maw](https://unsplash.com/@anthonymaw?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/a-car-that-has-crashed-into-another-car-XcjVef6uvYA?utm_source=autoclaim_adjudicator&utm_medium=referral) | a car that has crashed into another car |
| 2 | [Clark Van Der Beken](https://unsplash.com/@snapsbyclark?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/damaged-silver-car-with-crushed-hood-CSkriQWeTVs?utm_source=autoclaim_adjudicator&utm_medium=referral) | A damaged silver car with a crushed hood and broken headlight assembly |
| 3 | [Usman Malik](https://unsplash.com/@usmanbim94?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/a-red-car-is-on-a-flatbed-tow-truck-kE__1vnDxg4?utm_source=autoclaim_adjudicator&utm_medium=referral) | a red car is on a flatbed tow truck |
| 4 | [Usman Malik](https://unsplash.com/@usmanbim94?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/a-car-that-has-been-hit-by-another-car-xlxeGeh1DY4?utm_source=autoclaim_adjudicator&utm_medium=referral) | a car that has been hit by another car |
| 5 | [Scott Greer](https://unsplash.com/@sgreer?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/cars-are-involved-in-a-frontal-collision-RzVENp1UbRc?utm_source=autoclaim_adjudicator&utm_medium=referral) | Cars are involved in a frontal collision |
| 6 | [Josh Sonnenberg](https://unsplash.com/@joshsonnenberg?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/a-car-that-has-been-involved-in-a-car-accident-Tn9mxVvKL8g?utm_source=autoclaim_adjudicator&utm_medium=referral) | a car that has been involved in a car accident |
| 7 | [Odinei Ribeiro](https://unsplash.com/@odineiramone?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/a-car-that-is-sitting-on-the-side-of-the-road-UUGaMVsD63A?utm_source=autoclaim_adjudicator&utm_medium=referral) | a car that is sitting on the side of the road |
| 8 | [Karl Solano](https://unsplash.com/@karlsolano?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/black-car-on-brown-sand-during-daytime-WM5WRBl-EZ8?utm_source=autoclaim_adjudicator&utm_medium=referral) | black car on brown sand during daytime |
| 9 | [C Joyful](https://unsplash.com/@alonly?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/man-in-white-and-black-stripe-shirt-and-black-pants-standing-beside-black-car-during-daytime-uWOBgtCD_m8?utm_source=autoclaim_adjudicator&utm_medium=referral) | man in white and black stripe shirt and black pants standing beside bl |
| 10 | [Will Creswick](https://unsplash.com/@wilcre?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/blue-car-with-white-snow-on-top-YH1KID3mTpY?utm_source=autoclaim_adjudicator&utm_medium=referral) | blue car with white snow on top |
| 11 | [^_^](https://unsplash.com/@bionicdreamer?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/white-and-black-car-in-garage-qGzWty2oBlQ?utm_source=autoclaim_adjudicator&utm_medium=referral) | white and black car in garage |
| 12 | [Ante Hamersmit](https://unsplash.com/@ante_kante?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/parked-grey-car-pwhKQrvLZAQ?utm_source=autoclaim_adjudicator&utm_medium=referral) | parked grey car |
| 13 | [Josh Sonnenberg](https://unsplash.com/@joshsonnenberg?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/a-couple-of-cars-that-are-sitting-in-the-dirt-Bd79dpLgM6w?utm_source=autoclaim_adjudicator&utm_medium=referral) | a couple of cars that are sitting in the dirt |
| 14 | [Mr Brown](https://unsplash.com/@_mrbrown_?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/a-car-that-is-sitting-in-the-dirt-2hEqJKV6P1E?utm_source=autoclaim_adjudicator&utm_medium=referral) | a car that is sitting in the dirt |
| 15 | [Daniel](https://unsplash.com/@format_?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/a-car-that-is-sitting-in-the-grass-TXr-eYJUCKA?utm_source=autoclaim_adjudicator&utm_medium=referral) | A car that is sitting in the grass |
| 16 | [Bernd 📷 Dittrich](https://unsplash.com/@hdbernd?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/a-pick-up-truck-parked-in-the-middle-of-a-desert--93tHR17tWg?utm_source=autoclaim_adjudicator&utm_medium=referral) | a pick up truck parked in the middle of a desert |
| 17 | [Bernd 📷 Dittrich](https://unsplash.com/@hdbernd?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/a-close-up-of-a-tire-on-a-car-3jp8Sz3fYU4?utm_source=autoclaim_adjudicator&utm_medium=referral) | a close up of a tire on a car |
| 18 | [Brett Jordan](https://unsplash.com/@brett_jordan?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/overturned-car-with-police-vehicle-469n0Bm3rbU?utm_source=autoclaim_adjudicator&utm_medium=referral) | A black car overturned on a road beside a silver car and police vehicl |
| 19 | [Jonny Clow](https://unsplash.com/@jonnyclow?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/a-toy-truck-lies-overturned-in-a-shallow-stream-OQMe_62CGxE?utm_source=autoclaim_adjudicator&utm_medium=referral) | A toy truck lies overturned in a shallow stream |
| 20 | [Atiq Uz Zaman Khan](https://unsplash.com/@atiq_uz_zaman_khan?utm_source=autoclaim_adjudicator&utm_medium=referral) | [Unsplash](https://unsplash.com/photos/abandoned-car-with-missing-wheels-in-overgrown-lot-smALnx4U7_4?utm_source=autoclaim_adjudicator&utm_medium=referral) | Abandoned car with missing wheels in overgrown lot |

## Data shown in the UI

Crash stories on the Try a claim page: NHTSA vehicle-safety complaints, public domain (DATA.md #13).
