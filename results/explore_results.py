# +
import pandas as pd

pd.set_option('display.max_rows', df.shape[0]+1)
# -

df = pd.read_json('/mmfs1/gscratch/zlab/margsli/gitfiles/mod/results/hp_sweep_results.jsonl', lines=True)
print(df.groupby(['num_steps', 'lr','dense', 'in_domain']).ppl.mean())




