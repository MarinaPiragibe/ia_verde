# ia_verde_cnn

## Autora: Marina Piragibe

A estrutura do projeto é:

- modules:
    - cifake.py
    - encoders.py 
    - globals.py
    - termometro.py
    - utils.py
- feature_extraction.ipynb
- main_hdc.ipynb
- main_wisard.ipynb
- resnet18_cifake_finetuned_float32.pth

Na pasta modules estão contidas as classes e funções para execução de cada abordagem. Para o HDC há a classe Cifake que está em cifake.py e a codificação dos RecordEncoder em encoders.py. Já para a WiSARD a codificação dos atributos é feita pela função codificador_termometro em termômetro.py. Em globals.py estão contidas as definições de hiperparâmetros e preferências de execução. Por fim, utils.py possui funções auxiliares como para plotar imagens.

A extração de características do modelo pré-treinado está detalahada em feature_extraction.ipynb.

Toda a lógica de pré-processamento e execução de cada abordagem do modelo está detalhada em nos arquivos ipynb da raiz do projeto, sendo o main_hdc.ipynb para HDC e main_wisard.ipynb para a WiSARD. Os arquivos também contém a análise dos resultados com diversas métricas e as conclusões.

