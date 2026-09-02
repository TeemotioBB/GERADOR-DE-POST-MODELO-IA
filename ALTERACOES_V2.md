# V2 — operação de vídeos em 2 takes

## Regra do formato

Esta versão foi especializada para o padrão:

- todo vídeo normal possui 2 takes;
- take 1 = foto/vídeo + texto fixo na tela;
- take 2 = vídeo de continuação relacionado ao texto;
- a nova mídia substitui o take 1;
- o texto detectado no take 1 é recriado sobre a nova mídia;
- o áudio e o take 2 do original são preservados.

## Correções principais

### 1. OCR por linha e posição
O OCR antigo transformava todas as linhas encontradas no quadro em um texto único. A V2 mantém cada linha com sua posição (x/y/largura/altura).

### 2. Consenso temporal
Uma linha só é considerada candidata forte quando reaparece em vários frames do take 1 e na mesma região da tela.

### 3. Rejeição de texto aleatório
Logo, marca d'água, camiseta, placa e leituras isoladas recebem menos peso. Quando não existe consenso, o sistema deixa o campo vazio em vez de inventar uma legenda.

### 4. Segunda leitura localizada
Depois de descobrir onde está a legenda, a V2 faz uma nova leitura somente naquela região. Isso ajuda a recuperar primeira/última letra que o OCR global possa ter perdido.

### 5. Menos corte nas bordas
Defaults de crop alterados de 5%/8%/1% (topo/baixo/lado) para 1%/3%/0%, reduzindo o risco de amputar uma legenda perto da borda.

### 6. Detecção de take preservando cor
A análise de transição agora mantém informação de cor. O detector anterior convertia o frame para cinza antes da comparação.

### 7. Falha segura para o formato de 2 takes
No modo automático, se a troca não for encontrada com segurança, o sistema NÃO assume mais que o vídeo inteiro é take 1. Ele orienta a informar manualmente o segundo da troca.

## Validação feita

Foi executado um teste sintético 9:16 com:

- take 1 em movimento;
- frase fixa em duas linhas;
- texto de cenário em movimento;
- marca d'água fixa;
- take 2 visualmente diferente.

Resultado esperado/obtido:

`vem aqui hoje tomar um vinho cmg, não vai rolar nada`

A troca foi encontrada no ponto programado do teste (3,30 s).

## Arquivos que realmente mudaram

- `app.py`
- `core/captions.py`
- `core/config.py`
- `core/transition.py`
- `.env.example`
- `README.md`

Os demais foram mantidos para entregar um pacote completo e pronto para substituir no repositório.
