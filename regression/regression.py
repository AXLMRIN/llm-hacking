import pandas as pd 
import statsmodels.api as sm

def perform_regression(df: pd.DataFrame, y_column: str, x_column: str
) -> dict[str:str|float|list[float]]:
    """
    Perform logit regression and returns key values
    """
    Y = df[y_column].to_numpy().astype(int)
    X = df[x_column].to_numpy().astype(int)
    X = sm.add_constant(X)
    try: 
        model = sm.Logit(Y,X)
        res = model.fit()

        return {
            "Pseudo R-squared": res.prsquared,
            "Covariate Names": ["const", "x1"], # FIX is there a best way of doing this? 
            "Coef": res.params.tolist(),
            "Std err": res.bse.tolist(),
            "z": res.tvalues.tolist(),          # z-statistics
            "pvalues": res.pvalues.tolist(),
            "Conf Int": res.conf_int().tolist(),
            "Log-Likelihood": res.llf,
            "LL-Null": res.llnull,
            "LLR p-value": res.llr_pvalue,
            "AIC": res.aic,
            "BIC": res.bic,
            "N obs": res.nobs,
        }
    except: 
        return {
            "Pseudo R-squared": "FAILED",
            "Covariate Names": "FAILED", # FIX is there a best way of doing this? 
            "Coef": "FAILED",
            "Std err": "FAILED",
            "z": "FAILED",          # z-statistics
            "pvalues": "FAILED",
            "Conf Int": "FAILED",
            "Log-Likelihood": "FAILED",
            "LL-Null": "FAILED",
            "LLR p-value": "FAILED",
            "AIC": "FAILED",
            "BIC": "FAILED",
            "N obs": "FAILED",
        } 