import numpy as np


def codificador_termometro(data_series, n_bits):
    valor_min = data_series.min()
    valor_max = data_series.max()

    if valor_min == valor_max: # Lidar com atributos constantes
        return np.array([[0] * n_bits] * len(data_series), dtype=np.uint8)

    vetor_codificado = []
    for val in data_series:
        # Normaliza o valor para o range [0, n_bits - epsilon]
        valor_normalizado = (val - valor_min) / (valor_max - valor_min) * (n_bits - 1e-9)
        # Cria o vetor binário do termômetro
        thermometer_vec = [1 if j < valor_normalizado else 0 for j in range(n_bits)]
        vetor_codificado.append(thermometer_vec)
        #print(f"{val} -> {thermometer_vec}")
    return np.array(vetor_codificado, dtype=np.uint8)


#codificador_termometro(np.array([0,6,11]),6)