#!/usr/bin/env python
# coding: utf-8

# In[14]:


import math

import numpy as np
import pandas as pd
from scipy.stats import norm


# In[15]:


pd.read_csv("https://epoch.ai/data/all_ai_models.csv")


# # Function definitions

# In[16]:


def get_rank(
    df: pd.DataFrame,
    n: int | None = None,
    sort_col: str = "Publication date",
    val_col: str = "Training compute (FLOP)",
) -> pd.Series:
    """
    Cumulative rank of *val_col* up to each row, ordered by *sort_col*,
    robust to missing values.

    • If *val_col* is NaN for a row → rank is NaN.
    • Rows whose *val_col* is NaN do **not** affect later ranks.
    • Rows whose *sort_col* is NaN are treated as having unknown release time
      → their own rank is NaN and they do not affect others.
    • If *n* is given, ranks > n are set to NaN (frontier filter).

    Returns
    -------
    pd.Series aligned with *df.index* (dtype float, so NaNs are allowed).
    """
    # Sort chronologically; keep a stable sort to preserve original order ties
    ordered = df.sort_values(
        sort_col, kind="mergesort", na_position="last"
    ).reset_index()

    vals  = ordered[val_col]
    ranks = pd.Series(np.nan, index=ordered.index, dtype=float)

    # Working array of non-NaN values we have seen so far
    seen = []

    for idx, v in enumerate(vals):
        if pd.isna(v):           # current value is NaN → leave rank as NaN
            continue
        # Count how many previous non-NaN values are strictly larger
        rank = 1 + sum(prev > v for prev in seen)
        ranks.iloc[idx] = rank
        seen.append(v)           # add current value for future rows

    if n is not None:
        ranks = ranks.where(ranks <= n)

    # Re-align to the original DataFrame’s index order
    ranks.index = ordered["index"]
    return ranks.reindex(df.index)

def check_statistical_diff(row_open: pd.Series, row_closed: pd.Series, alpha: float = 0.05) -> bool:
    """
    H0: eci_open == eci_closed
    Two-sided z-test using combined SE = sqrt(s1^2 + s2^2).
    Returns True if |diff| is significant at level alpha, else False.
    Assumes row_* have keys 'eci' and 'eci_std'.
    """
    m1 = row_open.get('eci')
    m2 = row_closed.get('eci')
    s1 = row_open.get('eci_std')
    s2 = row_closed.get('eci_std')

    # Robust to missing values
    if any(pd.isna(x) for x in (m1, m2, s1, s2)):
        return False

    se = math.sqrt(s1**2 + s2**2)
    if se == 0:
        return abs(m1 - m2) > 0  # degenerate case

    z = abs(m1 - m2) / se
    zcrit = float(norm.ppf(1 - alpha/2))

    return z > zcrit


# # Setup

# In[18]:


pd.read_csv("https://epoch.ai/data/eci_scores.csv")


# In[ ]:


df = pd.read_csv("https://epoch.ai/data/eci_scores.csv")

# Featurization

# Dates
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Group model accessibility
df['Open'] = df['Model accessibility'].str.contains('Open weights')

# Get model rankings for each group
df_open = df[df['Open']].copy()
df_closed = df[~df['Open']].copy()

df_open['group_rank'] = get_rank(df_open, sort_col='date', val_col='eci')
df_closed['group_rank'] = get_rank(df_closed, sort_col='date', val_col='eci')

# Combine
df = pd.concat([df_open, df_closed]).reset_index().sort_values('date')

# Filter to top-1
df = df[df['group_rank'] <= 1]


# # Vertical gap

# In[12]:


# Get first and last dates with valid "vertical gap"
start_date = max(df[df['Open']]['date'].min(), df[~df['Open']]['date'].min())
end_date = min(df[df['Open']]['date'].max(), df[~df['Open']]['date'].max())

vertical_gaps = []

# For each day in range
for cur_date in pd.date_range(start_date, end_date):
    # Calculate the gap in eci
    best_open = df[(df['date'] <= cur_date) & (df['Open'])]['eci'].max()
    best_closed = df[(df['date'] <= cur_date) & (~df['Open'])]['eci'].max()
    gap = best_closed - best_open
    vertical_gaps.append((cur_date, gap))

# Convert to DataFrame
vertical_gaps_df = pd.DataFrame(vertical_gaps, columns=['date', 'vertical_gap'])
vertical_gap_ci_90 = vertical_gaps_df['vertical_gap'].quantile([0.05, 0.95]).values
print(f"Average vertical gap: {vertical_gaps_df['vertical_gap'].mean():.2}")
print(f"Standard deviation: {vertical_gaps_df['vertical_gap'].std():.2}")
print(f"90% confidence interval: {vertical_gap_ci_90[0]:.1f} to {vertical_gap_ci_90[1]:.1f}")


# # Horizontal gap
# Here we have to do a bit more work, since we have uncertainty over the ECI score. We want to treat as "colliding" any cases where the difference between ECI scores are not statistically significant.

# In[13]:


# Get first and last eci scores with valid "horizontal gap"
start_eci = max(df[df['Open']]['eci'].min(), df[~df['Open']]['eci'].min())
end_eci = min(df[df['Open']]['eci'].max(), df[~df['Open']]['eci'].max())

horizontal_gaps = []

# We'll keep track of which open models might still qualify as a colission to reduce duplicated computations
df_open_possible = df[df['Open']].copy()

# For each eci in range
for cur_eci in np.linspace(start_eci, end_eci, 100):
    # Find date of first open-source model where ECI is not statistically different from earliest closed-source model
    # with ECI >= cur_eci
    cur_closed_model = df[(df['eci'] >= cur_eci) & (~df['Open'])].sort_values('date').iloc[0]
    cur_open_model = None
    # Check through open models, starting earliest to latest
    for _, row in df_open_possible.iterrows():
        if row['eci'] < cur_eci:
            is_statistically_lower = check_statistical_diff(row, cur_closed_model)
            if not is_statistically_lower:
                cur_open_model = row
                gap = cur_open_model['date'] - cur_closed_model['date']
                horizontal_gaps.append((cur_eci, gap.days / 30.5))
                break
            else:
                df_open_possible = df_open_possible[df_open_possible['date'] > row['date']]
        else:
            gap = row['date'] - cur_closed_model['date']
            horizontal_gaps.append((cur_eci, gap.days / 30.5))
            break

# Convert to DataFrame
horizontal_gaps_df = pd.DataFrame(horizontal_gaps, columns=['eci', 'horizontal_gap'])
horizontal_gap_ci_90 = horizontal_gaps_df['horizontal_gap'].quantile([0.05, 0.95]).values
print(f"Average horizontal gap: {horizontal_gaps_df['horizontal_gap'].mean():.2f} months")
print(f"Standard deviation: {horizontal_gaps_df['horizontal_gap'].std():.1f}")
print(f"90% confidence interval: {horizontal_gap_ci_90[0]:.1f} to {horizontal_gap_ci_90[1]:.1f}")


# In[ ]:




