# 📡 Protocolo W.P.A: Riemann Radar

> **"Uma abordagem computacional para decodificar a 'música' dos números primos."**

O **Protocolo W.P.A** (*William Pereira de Almeida*) é um sistema experimental que combina Teoria Analítica dos Números com Machine Learning para prever a localização de números primos em acesso aleatório, superando a precisão das aproximações logarítmicas tradicionais.

---

## 🎯 O Objetivo
Desafiar a noção de que a distribuição dos números primos é puramente caótica e imprevisível localmente.

O objetivo foi criar um algoritmo capaz de:
1. Receber um índice $n$ (ex: "Quero o 100.000º primo").
2. Calcular sua posição sem precisar iterar pelos antecessores.
3. Obter uma precisão alta o suficiente para tornar a busca trivial.

## 🧠 A Tecnologia: Hibridismo Físico-Matemático

O **Riemann Radar** opera em três camadas:

### 1. Camada Base: Expansão de Cipolla
Utilizamos uma regressão linear sobre 7 termos da expansão de Cipolla para criar a "estrada principal" da distribuição dos primos.
> *Erro Médio Base: ~111 unidades*

### 2. Camada Espectral: Zeros de Riemann
Aqui reside a inovação. O sistema calcula 5.000 ondas senoidais baseadas nos **Zeros Não-Triviais da Função Zeta de Riemann**. Utilizamos regressão `Ridge` para sintonizar a fase e amplitude dessas ondas, criando uma interferência construtiva que prevê as oscilações do erro.
> *Erro Médio com Radar: ~20 unidades*

### 3. Camada Neural: Correção Não-Linear
Uma Rede Neural (MLP Regressor) analisa os resíduos que a física linear não conseguiu explicar, refinando a previsão final para níveis de precisão de dois dígitos.

---

## 📊 Resultados Obtidos

Durante os testes de estresse (0 a 100.000 primos):

| Modelo | Tecnologia | Erro Médio (MAE) | Precisão Relativa |
| :--- | :--- | :--- | :--- |
| **Teoria Padrão** | Fórmula $n \ln n$ | ~460.00 | 99.92% |
| **Protocolo W.P.A** | **Riemann Radar** | **~20.96** | **99.998%** |

O sistema foi capaz de localizar primos na casa dos 1.3 milhões com um tempo de inferência de **0.006 segundos**.

---

## 🛠️ Como Usar

1. Clone o repositório.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt



## 📝 Autor

*William Pereira de Almeida* Desenvolvido em: Novembro de 2025
Projeto de Investigação em Machine Learning & Teoria dos Números
