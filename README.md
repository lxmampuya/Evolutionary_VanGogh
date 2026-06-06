# Evolutionary Van Gogh

To run this code, install the required packages from requirements.txt and open the Jupyter notebook (`analysis.ipynb`) for an example of how to run the code and experiments with it. The code was tested with Python version 3.9.5, more recent versions may need changes to the requirements.

```bash
pip install -r requirements.txt
```

To start a Jupyter notebook instance, have Jupyter notebook [installed](https://jupyter.org/install#jupyter-notebook) and start it up in this directory.

---

## Hypothesis1
- Hierarchical EA where we start with fewer points in earlier generation and increase it later. See hier_evo.py for code changes.
1. Switch to the branch `git switch hypo1-hierarchical`
2. Open the notebook analysis_h1.ipynb. You may load the saved data (df object) with `df = pd.read_pickle("0604_evolution_results.pkl")`. This is also in a cell near the bottom of the notebook.
3. Re-run the cells if desired to see training process, it takes 1-2 minutes to run each.
4. The generated images results are saved in img/h1/. The numbers of each folder correspond to the experiments numbered in the notebook.

---

#Hypothesis 2
- Mutation Decay for both the mutation strength and the mutation probability
1. switch to the branch `git switch decay-mutation-rate`
2. the full experiment is in `analysis.ipynb` including the figures used in the presentation
3. If you wish to also run the random search you can change `random_search = False` into `random_search = True` in the third cell and run the notebook again

---

