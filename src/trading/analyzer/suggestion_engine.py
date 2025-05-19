from durable.engine import ruleset

class SuggestionEngine:
    def __init__(self, config: Dict):
        self.config = config

    def generate_suggestions(self, stats: Dict, df: pd.DataFrame) -> List[str]:
        suggestions = []
        with ruleset("suggestions"):
            for rule in self.config["suggestions"]:
                if eval(rule["condition"], {"stats": stats, "df": df}):
                    suggestions.append(rule["action"])
        return suggestions