# Start here

Everything you need to begin your part. Takes about 10 minutes.

## 1. Set up (once)

```bash
git clone https://github.com/Manavpatel06/sml-signal-or-noise.git
cd sml-signal-or-noise
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_data.py        # ~34 MB into data/raw/, never committed
pytest -q                              # should pass, with your tests skipped
```

## 2. Make your branch

```bash
git checkout main
git pull
git checkout -b <your-branch>
```

| Who | Branch | Your instructions |
| --- | --- | --- |
| Rashi | `role2-knn-kmeans` | `.ai/roles/2-rashi-knn-kmeans.md` |
| Karthik | `role3-pca-hmm` | `.ai/roles/3-karthik-pca-hmm.md` |
| Harsh | `role4-gmm-em` | `.ai/roles/4-harsh-gmm-em.md` |
| Tejas | `role5-discrimination` | `.ai/roles/5-tejas-discrimination.md` |
| Ziyad | `role6-evaluation` | `.ai/roles/6-ziyad-evaluation.md` |

(`.ai/` is a hidden folder. On Windows, turn on "Hidden items" in File Explorer's View menu, or just open it from your editor.)

## 3. Paste this into your coding agent

Open the repo folder in your agent, then send this as the first message,
replacing the role file with yours:

```
You are working in this repository on a graded course project.
Read these files first, in order, and follow them exactly:
  1. .ai/AGENTS.md            (shared rules for everyone)
  2. .ai/roles/<MY ROLE FILE> (my tasks)
  3. README.md
Then implement my tasks one at a time, in the order listed.
After each task run `pytest -q` and fix anything that fails.
Do not change shared interfaces or other people's files.
Do not use scikit-learn, hmmlearn or similar libraries for the algorithms.
When a task is done, summarise what changed and the test results.
```

Read and understand every line your agent writes. We have to explain our code
in the presentation Q&A, and the course requires our own implementations.

## 4. Hand in your work

```bash
pytest -q                                   # must pass
python scripts/run_bench.py --detectors ksigma <yours>   # detectors only
git add -A
git commit -m "Short description of the change"
git push -u origin <your-branch>
```

Open a pull request into `main` on GitHub. Paste the bench summary table into
the description. Your review partner approves it before it's merged:
Rashi ↔ Karthik, Harsh ↔ Tejas, Manav ↔ Ziyad.

## Deadlines

| Date | What |
| --- | --- |
| **Sun Nov 9** | First working version of your part, PR open |
| Nov 15 | All results in |
| Nov 16–22 | Slides, dry run |
| Nov 23 – Dec 2 | Presentation |
| Dec 7 | Final report |

Questions or blocked on something: post in the group chat right away.
