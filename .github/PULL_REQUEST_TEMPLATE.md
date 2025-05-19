# -*- coding: utf-8 -*-
# MIA_IA_SYSTEM_v2_2025/.github/PULL_REQUEST_TEMPLATE.md
# Modèle de Pull Request pour MIA_IA_SYSTEM_v2_2025
#
# Version: 2.1.5
# Date: 2025-05-14
#
# Rôle: Standardise les pull requests pour faciliter les revues de code, incluant des exigences
#       de tests pour les nouveaux fichiers (risk_manager.py, regime_detector.py, feature_pipeline.py, trade_probability.py).
#
# Utilisé par: GitHub pour structurer les PRs manuelles et automatisées (Dependabot).
#
# Notes:
# - Conforme à structure.txt (version 2.1.5, 2025-05-14).
# - Supporte les suggestions :
#   - 1 (Position sizing dynamique) : Tests pour risk_manager.py.
#   - 4 (HMM/Changepoint Detection) : Tests pour regime_detector.py.
#   - 7 (Tests unitaires, couverture 100%).
# - Intègre avec .github/workflows/python.yml, .pre-commit-config.yaml.
# - Inclut des exigences pour feature_pipeline.py, trade_probability.py.
# - Pas de références à dxFeed, obs_t, 320 features, ou 81 features.
# - Policies Note: The official directory for routing policies is src/model/router/policies.
#   The src/model/policies directory is a residual and should be verified for removal to avoid import conflicts.

## Description
Décrivez les changements apportés dans cette pull request, y compris :
- Le problème résolu ou la fonctionnalité ajoutée.
- Les fichiers modifiés et leur rôle (ex. : `src/model/trade_probability.py` pour le réentraînement).
- Toute information contextuelle pertinente (ex. : lien vers une issue GitHub).

## Related Improvements
Cochez les améliorations liées à cette PR :
- [ ] Position sizing dynamique (suggestion 1)
- [ ] HMM / Changepoint Detection (suggestion 4)
- [ ] Safe RL / CVaR-PPO (suggestion 7)
- [ ] Distributional RL / QR-DQN (suggestion 8)
- [ ] Ensembles de politiques (suggestion 10)
- [ ] Other (specify)

## Files Modified
Cochez les fichiers modifiés :
- [ ] src/risk_management/risk_manager.py
- [ ] src/features/regime_detector.py
- [ ] src/features/feature_pipeline.py
- [ ] src/model/trade_probability.py
- [ ] Other (specify)

## Tests
Cochez les tests ajoutés ou mis à jour :
- [ ] Tests added/updated in `tests/test_risk_manager.py`
- [ ] Tests added/updated in `tests/test_regime_detector.py`
- [ ] Tests added/updated in `tests/test_feature_pipeline.py`
- [ ] Tests added/updated in `tests/test_trade_probability.py`
- [ ] Tests passed (`pytest tests/ -v`)

## Checklist
- [ ] Les tests unitaires passent (`pytest tests/ -v`).
- [ ] La couverture de code est ≥ 100% (`pytest --cov=src --cov-report=html --cov-fail-under=100`).
- [ ] Les tests de résilience passent (`pytest tests/test_resilience.py -v`).
- [ ] Le code est formaté avec Black et isort (`pre-commit run --all-files`).
- [ ] Le linting est réussi avec Flake8 et MyPy (`flake8 src/ tests/ scripts/`, `mypy src/`).
- [ ] La documentation est mise à jour (ex. : `docs/modules.md`, `docs/api_reference.md`).
- [ ] Les changements respectent les standards de `structure.txt` (version 2.1.5).
- [ ] Aucun code obsolète (ex. : dxFeed, obs_t, 320/81 features).

## Notes supplémentaires
Ajoutez toute information complémentaire, comme :
- Dépendances modifiées dans `requirements.txt`.
- Impact sur les autres modules (ex. : `MiaSwitcher`, `TradeProbability`).
- Instructions spécifiques pour les reviewers.