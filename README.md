# 1o1_OLS_TEMPLATE

I am quite new to this data modelling and have been working on kaggle notebook and i wanted to make myself a template to follow, let me know if i am missing something or if some stuff could be done differently !

steps for a linear regression into regularization

observe Y

python
import matplotlib.pyplot as plt
import seaborn as sns

df['Y'].describe()


# Clean data to remove infinities and NaNs
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Y'])
sns.displot(df['Y'], kde=True)
observe features relationships

python
import matplotlib.pyplot as plt
import pandas as pd

# Select your features
cols = [
    'X_num1', 'X_num2', 'X_num3', 'X_num4',
    'X_num5', 'X_num6', 'X_num7', 'X_num8',
    'X_oh1', 'X_oh2', 'X_ord1'
]

# --- Plot pairwise scatterplots + histograms (diagonal) ---
pd.plotting.scatter_matrix(
    df[cols],
    figsize=(14, 10),
    diagonal='hist',      # or 'kde' for density on diagonal
    alpha=0.6,
    color='steelblue',
    edgecolor='white'
)

# Adjust layout
plt.suptitle("Pairwise Feature Relationships", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()
encode categorical variables

python
# --- 1) Encode ordinal variable X_ord1 ---
# Only map if it's still strings (object); if already numeric, this will be skipped
if df['X_ord1'].dtype == 'O':
    ord_map = {'Bearish': 0, 'Neutral': 1, 'Bullish': 2}
    df['X_ord1'] = df['X_ord1'].map(ord_map)
python
# --- 2) One-hot encode nominal variables X_oh1 and X_oh2 ---
oh_source_cols = ['X_oh1', 'X_oh2']
df_oh = pd.get_dummies(df, columns=oh_source_cols, drop_first=True)
df_oh = df_oh.astype(int)
python
# --- 3) Order columns neatly (optional) ---
num_cols = [f'X_num{i}' for i in range(1, 9)]
# Get all new dummy columns automatically
oh_cols = [c for c in df_oh.columns if c.startswith('X_oh1_') or c.startswith('X_oh2_')]
ord_cols = ['X_ord1']
target = ['Stock_Price']
python
ordered_cols = num_cols + oh_cols + ord_cols + target
ordered_cols = [c for c in ordered_cols if c in df_oh.columns]
df_final = df_oh[ordered_cols].copy()
Check correlation of Xs

python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assume df_final is your preprocessed DataFrame with X features only
X_cols = [c for c in df_final.columns if c.startswith(('X_num', 'X_oh', 'X_ord'))]
corr_matrix = df_final[X_cols].corr(method='pearson')

# Plot
fig, ax = plt.subplots(figsize=(10,8))
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Correlation", rotation=270, labelpad=15)

# Label axes
ax.set_xticks(np.arange(len(X_cols)))
ax.set_yticks(np.arange(len(X_cols)))
ax.set_xticklabels(X_cols, rotation=90)
ax.set_yticklabels(X_cols)

# Annotate correlation values
for i in range(len(X_cols)):
    for j in range(len(X_cols)):
        value = corr_matrix.iloc[i, j]
        # choose text color based on background brightness for readability
        color = "white" if abs(value) > 0.5 else "black"
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

plt.title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.show()
Train-test split and transformation

python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- 0) Working copy ----------
df_model = df_final.copy()  # your encoded dataframe


# ---------- 1) Identify columns ----------
target_col = 'Stock_Price' if 'Stock_Price' in df_model.columns else 'Y'
num_cols = [c for c in df_model.columns if c.startswith('X_num')]
oh_cols  = [c for c in df_model.columns if c.startswith('X_oh')]
ord_cols = ['X_ord1'] if 'X_ord1' in df_model.columns else []

# Ensure dummies are numeric 0/1
df_model[oh_cols] = df_model[oh_cols].astype(int)

# ---------- 2) Train / test split ----------
X = df_model.drop(columns=[target_col])
y = df_model[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assume df_model is your working dataframe
num_cols = [c for c in df_model.columns if c.startswith('X_num')]

# Compute skewness
skews = df_model[num_cols].skew(numeric_only=True).sort_values(ascending=False)
print("Skewness per numeric feature:\n", skews, "\n")

# Create subplots
rows = int(np.ceil(len(num_cols) / 3))
fig, axes = plt.subplots(rows, 3, figsize=(16, 4 * rows))
axes = axes.flatten()

# Plot each numeric feature
for i, col in enumerate(num_cols):
    ax = axes[i]
    ax.hist(df_model[col], bins=30, color='steelblue', edgecolor='white', alpha=0.8, density=True)
    ax.set_title(f"{col}\nSkew: {skews[col]:.2f}")
    ax.set_xlabel("")
    ax.set_ylabel("Density")

# Hide empty subplots if any
for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Distributions of Numeric Features (Raw)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
python
# Heuristic: log1p if |skew| > 0.75 and strictly positive
log_cols   = [c for c in num_cols if abs(skews[c]) > 0.75 and (X_train[c] > 0).all()]
plain_cols = [c for c in num_cols if c not in log_cols]
python
# ---------- 4) Apply log1p to TRAIN numeric (inplace on copies) ----------
X_train_log = X_train.copy()
for c in log_cols:
    X_train_log[c] = np.log1p(X_train_log[c])
python
# Apply the SAME transform to TEST
X_test_log = X_test.copy()
for c in log_cols:
    X_test_log[c] = np.log1p(X_test_log[c])
python
# ---------- 5) Standardize numeric features ----------
scaler = StandardScaler()
scaled_train = pd.DataFrame(
    scaler.fit_transform(X_train_log[num_cols]),
    columns=num_cols, index=X_train_log.index)
scaled_test = pd.DataFrame(
    scaler.transform(X_test_log[num_cols]),
    columns=num_cols, index=X_test_log.index)

X_train_log[num_cols] = scaled_train
X_test_log[num_cols] = scaled_test
python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assume df_model is your working dataframe
num_cols = [c for c in X_train_log.columns if c.startswith('X_num')]

# Compute skewness
skews = X_train_log[num_cols].skew(numeric_only=True).sort_values(ascending=False)
print("Skewness per numeric feature:\n", skews, "\n")

# Create subplots
rows = int(np.ceil(len(num_cols) / 3))
fig, axes = plt.subplots(rows, 3, figsize=(16, 4 * rows))
axes = axes.flatten()

# Plot each numeric feature
for i, col in enumerate(num_cols):
    ax = axes[i]
    ax.hist(X_train_log[col], bins=30, color='steelblue', edgecolor='white', alpha=0.8, density=True)
    ax.set_title(f"{col}\nSkew: {skews[col]:.2f}")
    ax.set_xlabel("")
    ax.set_ylabel("Density")

# Hide empty subplots if any
for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Distributions of Numeric Features (Raw)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
python
# ---------- 6) Reassemble final frames (order optional) ----------
ordered_cols = num_cols + oh_cols + ord_cols
ordered_cols = [c for c in ordered_cols if c in X_train_log.columns]

X_train_scaled = X_train_log[ordered_cols].copy()
X_test_scaled  = X_test_log[ordered_cols].copy()

# ---------- 7) Sanity checks ----------
print("Skew on train numeric features:")
print(skews.sort_values(ascending=False), "\n")

print("Log-transformed numeric columns:", log_cols)
print("Plain-scaled numeric columns:", plain_cols, "\n")

print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
print("First 5 cols:", X_train_scaled.columns[:5].tolist())
fit the lin reg

python
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# -------- Prepare data --------
X_train_sm = sm.add_constant(X_train_scaled)   # adds intercept term
X_test_sm  = sm.add_constant(X_test_scaled)

# Fit OLS model
ols_model = sm.OLS(y_train, X_train_sm).fit()

# Predictions
y_pred = ols_model.predict(X_test_sm)

# -------- Model summary --------
print(ols_model.summary())
python
mse_train = mean_squared_error(y_train, ols_model.predict(X_train_sm))
mse_test  = mean_squared_error(y_test, y_pred)

print(f"Train MSE: {mse_train:.3f}")
print(f"Test  MSE: {mse_test:.3f}")
Check Linear Regression Assumptions

(A) Linearity Residuals should not show a pattern versus fitted values.

python
import matplotlib.pyplot as plt

residuals = y_train - ols_model.fittedvalues
fitted = ols_model.fittedvalues

plt.figure(figsize=(6,4))
plt.scatter(fitted, residuals, alpha=0.7, color='steelblue', edgecolor='white')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values (Linearity Check)")
plt.show()
(B) Normality of residuals : Residuals should follow a normal distribution. p > 0.05 → residuals not significantly different from normal

python
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=30, color='steelblue', edgecolor='white', density=True, alpha=0.8)
plt.xlabel("Residuals")
plt.ylabel("Density")
plt.title("Residuals Distribution (Normality Check)")
plt.show()
(C) Homoscedasticity (constant variance) p > 0.05 → homoscedasticity holds. p < 0.05 → heteroscedasticity (variance changes with fitted values).

It plots:

X-axis → theoretical quantiles from a normal distribution

Y-axis → quantiles of your actual residuals

The red (or gray) 45° line represents perfect normality. If your residuals are normally distributed, their quantiles should match those of a normal distribution → all points should lie close to that line.

python
import statsmodels.api as sm
sm.qqplot(residuals, line='45', fit=True)
plt.title("Q–Q Plot of Residuals")
plt.show()
(D) Independence of errors

Use the Durbin–Watson statistic (printed in model summary).

Rule of thumb:

~2 → no autocorrelation

<1.5 → positive autocorrelation

2.5 → negative autocorrelation

python
plt.figure(figsize=(6,4))
plt.scatter(fitted, residuals, color='steelblue', alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Check for Homoscedasticity")
plt.show()
(F) Influential observations

Check for outliers that heavily influence the regression fit.

✅ Most points have Cook’s distance < 1. ❌ Points above 1 are influential — consider investigating them.

python
import matplotlib.pyplot as plt
import numpy as np

# --- Compute Cook's distances ---
influence = ols_model.get_influence()
c, _ = influence.cooks_distance

# --- Find top influential observations ---
n_to_label = 5  # number of points to label
top_idx = np.argsort(c)[-n_to_label:]  # indices of top 5 highest Cook’s distances

# --- Plot Cook’s Distance ---
plt.figure(figsize=(10,5))
markerline, stemlines, baseline = plt.stem(range(len(c)), c, markerfmt=",", basefmt=" ")
plt.setp(markerline, color='steelblue', alpha=0.7)
plt.setp(stemlines, color='steelblue', alpha=0.5)

plt.axhline(1, color='red', linestyle='--', linewidth=1)
plt.xlabel("Observation Index")
plt.ylabel("Cook’s Distance")
plt.title("Influential Observations (Cook’s Distance)")

# --- Label top influential points ---
for i in top_idx:
    plt.annotate(
        str(i), 
        xy=(i, c[i]), 
        xytext=(i, c[i] + 0.02),  # small vertical offset
        textcoords="data",
        ha='center', 
        fontsize=9, 
        color='darkred',
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.7)
    )

plt.tight_layout()
plt.show()
python
# If you want to see their actual data values later:
df_model.iloc[top_idx]
Now Lasso

python
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ========= 1) Set up CV + parameter grid =========
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
    "alpha": np.logspace(-4, 2, 60),   # 1e-4 ... 1e2
    "max_iter": [10000]
    # You can add more if you want: "fit_intercept": [True, False]
}
python
# ========= 2) Grid search with CV over alpha =========
# Choose scoring: 'neg_mean_squared_error' or 'r2'
gs = GridSearchCV(
    estimator=Lasso(random_state=42),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',   # refit on best (lowest MSE)
    cv=kf,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)
gs.fit(X_train_scaled, y_train)

best_alpha = gs.best_params_["alpha"]
print(f"Best alpha (λ): {best_alpha:.6f}")
print(f"Best CV score (neg MSE): {gs.best_score_:.6f}")
python
# ========= 3) Refit model available as gs.best_estimator_ =========
lasso_best = gs.best_estimator_
python
# ========= 4) Train/Test performance =========
y_train_pred = lasso_best.predict(X_train_scaled)
y_test_pred  = lasso_best.predict(X_test_scaled)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test  = mean_squared_error(y_test, y_test_pred)
r2_train  = r2_score(y_train, y_train_pred)
r2_test   = r2_score(y_test, y_test_pred)

print(f"Train MSE: {mse_train:.4f} | Test MSE: {mse_test:.4f}")
print(f"Train R² : {r2_train:.4f} | Test R² : {r2_test:.4f}")
python
# ========= 5) Coefficients (sparsity) =========
coefs = pd.Series(lasso_best.coef_, index=X_train_scaled.columns, name="coef")
coefs_nonzero = coefs[coefs != 0].sort_values(key=np.abs, ascending=False)
print("\nNon-zero coefficients (sorted by |coef|):")
print(coefs_nonzero)
print(f"\nNumber of non-zero features: {np.sum(lasso_best.coef_ != 0)} / {len(lasso_best.coef_)}")
print(f"Intercept: {lasso_best.intercept_:.4f}")
python
# ========= 6) Plot CV curve: mean CV MSE vs alpha =========
# GridSearchCV cv_results_: means are over folds; note scoring is NEGATIVE MSE
results = pd.DataFrame(gs.cv_results_)
# Keep only rows varying over alpha (max_iter fixed)
results = results.sort_values("param_alpha")
alphas_sorted = results["param_alpha"].astype(float).values
mean_test_mse = -results["mean_test_score"].values  # negate back to MSE
std_test_mse  = results["std_test_score"].values

plt.figure(figsize=(7,4))
plt.plot(alphas_sorted, mean_test_mse, marker='o', linewidth=1, label='CV mean MSE')
plt.fill_between(alphas_sorted,
                 mean_test_mse - std_test_mse,
                 mean_test_mse + std_test_mse,
                 alpha=0.2, label='±1 std')
plt.axvline(best_alpha, color='red', linestyle='--', linewidth=1.2, label=f'best α = {best_alpha:.4f}')
plt.xscale('log')
plt.gca().invert_xaxis()  # small→large left→right if you prefer: comment out if not desired
plt.xlabel("alpha (log scale)")
plt.ylabel("CV Mean MSE")
plt.title("Lasso GridSearchCV: CV Mean MSE vs alpha")
plt.legend()
plt.tight_layout()
plt.show()
python
# ========= 7) Predicted vs Actual (with perfect-fit reference line) =========
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_test_pred, alpha=0.7, color='steelblue', edgecolor='white', label='Predicted vs Actual')

# Compute range for perfect fit line
min_y = float(np.min([y_test.min(), y_test_pred.min()]))
max_y = float(np.max([y_test.max(), y_test_pred.max()]))

# Perfect fit (y = x)
plt.plot([min_y, max_y], [min_y, max_y], color='red', linestyle='--', linewidth=2, label='Perfect Fit (y = x)')

# Optional: add best-fit line for predictions
coef = np.polyfit(y_test, y_test_pred, 1)
poly1d_fn = np.poly1d(coef)
plt.plot([min_y, max_y], poly1d_fn([min_y, max_y]), color='green', linestyle='-', linewidth=1.5, label='Model Fit Line')

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Lasso (best α) — Predicted vs Actual (Test Set)")
plt.legend()
plt.axis("equal")  # makes x and y scales identical
plt.tight_layout()
plt.show()
