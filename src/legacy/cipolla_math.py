"""
LEGADO: MODELO MATEMÁTICO PURO (CIPOLLA REGRESSION)
Data: 29/11/2025
Descrição: Tentativa de ajustar a Expansão de Cipolla (7 termos) aos primos
usando Mínimos Quadrados, sem uso de Inteligência Artificial.
"""

import numpy as np
import time

print("📐 INICIANDO MODELO LEGADO: CIPOLLA MATH...")

# 1. GERAR DADOS
limit = 80000
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

# 2. CONSTRUIR A MATRIZ DE CIPOLLA (7 TERMOS)
print("⚙️ Calculando termos algébricos...")
n_safe = np.maximum(n, 2.72)
ln = np.log(n_safe)
lnln = np.log(np.log(n_safe))
sqrt_n = np.sqrt(n_safe)

# Basis Functions
t1 = n_safe * ln
t2 = n_safe * lnln
t3 = n_safe
t4 = (n_safe * lnln) / ln
t5 = n_safe / ln
t6 = (n_safe * (lnln**2)) / ln
t7 = sqrt_n * ln

B = np.vstack([t1, t2, t3, t4, t5, t6, t7]).T

# 3. FIT (MÍNIMOS QUADRADOS)
print("🧮 Ajustando coeficientes...")
coeffs, _, _, _ = np.linalg.lstsq(B, real_primes, rcond=None)

# 4. AVALIAÇÃO
pred = B.dot(coeffs)
erro = np.abs(real_primes - pred)
mae = np.mean(erro)

print(f"\n✅ RESULTADO DO MODELO MATEMÁTICO:")
print(f"   Erro Médio Absoluto (MAE): {mae:.2f}")
print(f"   Coeficientes Encontrados: {coeffs}")

# Função para uso manual
def prever_cipolla(n_val):
    # Recalcula para um n específico
    ns = max(n_val, 2.72)
    l = np.log(ns)
    ll = np.log(np.log(ns))
    sq = np.sqrt(ns)
    terms = np.array([
        ns*l, ns*ll, ns, 
        (ns*ll)/l, ns/l, (ns*(ll**2))/l, 
        sq*l
    ])
    return int(np.dot(terms, coeffs))

print(f"\nTeste Prático (n=50.000): {prever_cipolla(50000)} (Real: {real_primes[49999]})")