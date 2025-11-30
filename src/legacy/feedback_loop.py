"""
LEGADO: O ORÁCULO COM FEEDBACK (XGBOOST AUTOREGRESSIVO)
Data: 29/11/2025
Descrição: Modelo que utiliza a previsão anterior para corrigir a próxima.
Tem a maior precisão de todas (Erro ~9), mas não serve para acesso aleatório
pois depende de saber onde você estava antes.
"""

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pandas as pd

print("🔄 INICIANDO MODELO LEGADO: FEEDBACK LOOP...")

# 1. DADOS
limit = 50000
print(f"📚 Carregando {limit} primos...")
primes = []
estimate_limit = int(limit * np.log(limit) * 1.3)
is_prime = np.ones(estimate_limit, dtype=bool)
is_prime[:2] = False
for i in range(2, int(estimate_limit**0.5) + 1):
    if is_prime[i]:
        is_prime[i*i::i] = False
for i, p in enumerate(is_prime):
    if p:
        primes.append(i)
        if len(primes) >= limit: break

real_primes = np.array(primes)
n = np.arange(1, limit + 1).astype(float)

# 2. ENGENHARIA DE FEATURES COM MEMÓRIA
# Fórmula Base (Simplificada para dar suporte à IA)
# Usando coeficientes aproximados da fase anterior
a, b, c, d = [1.1426, 2.1650, -0.9285, -1.6059]
n_safe = np.maximum(n, 2.72)
log_n = np.log(n_safe)
base_pred = (a * n_safe * log_n) + (b * n_safe) # Simplificado
target_ratio = real_primes / base_pred

# Criando Lags (Memória)
ratio_lag1 = np.roll(target_ratio, 1)
ratio_lag2 = np.roll(target_ratio, 2)
ratio_lag1[:1] = 1.0
ratio_lag2[:2] = 1.0

X = np.column_stack([
    n, np.log(n), 
    n % 210, n % 6, np.sin(n/10),
    ratio_lag1, ratio_lag2 # <--- O Segredo do Feedback
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. TREINAMENTO
print("🔥 Treinando IA com Memória...")
model = xgb.XGBRegressor(
    n_estimators=1000, 
    learning_rate=0.05, 
    max_depth=6,
    random_state=42,
    n_jobs=-1
)
model.fit(X_scaled, target_ratio)

# 4. FUNÇÃO GERADORA DE FUTURO
def gerar_sequencia(n_inicio, qtd_passos, last_ratios):
    print(f"\n🔮 Gerando sequência de {qtd_passos} primos a partir de {n_inicio}...")
    current_n = float(n_inicio)
    memory = list(last_ratios)
    preds = []
    
    for _ in range(qtd_passos):
        # Recriar features
        l_n = np.log(max(current_n, 2.72))
        feats = [
            current_n, l_n, 
            current_n % 210, current_n % 6, np.sin(current_n/10),
            memory[-1], memory[-2]
        ]
        
        # Prever Ratio
        x_in = scaler.transform([feats])
        pred_ratio = model.predict(x_in)[0]
        
        # Calcular Primo
        base = (a * current_n * l_n) + (b * current_n)
        est = int(base * pred_ratio)
        preds.append(est)
        
        # Atualizar memória (A IA confia nela mesma)
        memory.append(pred_ratio)
        current_n += 1
        
    return preds

# Teste
ultimos = target_ratio[-2:]
futuro = gerar_sequencia(limit + 1, 5, ultimos)
print(f"Próximos 5 primos estimados: {futuro}")