from delta.tables import DeltaTable
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

class ShapAnalyzer:
    def __init__(self, config: Dict):
        self.config = config

    def calculate_shap(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            model = DecisionTreeClassifier().fit(df.drop("reward", axis=1), df["reward"] > 0)
            pi = permutation_importance(model, df.drop("reward", axis=1), df["reward"] > 0, n_repeats=10)
            top_features = df.drop("reward", axis=1).columns[pi.importances_mean.argsort()[-100:]]
            shap_values = calculate_shap(df[top_features], target="reward", max_features=SHAP_FEATURE_LIMIT)
            filtered = self.filter_shap_features(shap_values, threshold=0.01)
            DeltaTable.forPath(BASE_DIR / "data/shap").update(
                updates={"values": filtered.to_dict()},
                predicate="date = current_date"
            )
            filtered.to_parquet(BASE_DIR / f"data/shap/{datetime.now().strftime('%Y%m%d')}.parquet")
            return filtered
        except Exception as e:
            logger.error(f"Erreur calcul SHAP: {str(e)}")
            return None

    def filter_shap_features(self, shap_df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        mean_abs = shap_df.abs().mean()
        return shap_df[mean_abs[mean_abs >= threshold].index]