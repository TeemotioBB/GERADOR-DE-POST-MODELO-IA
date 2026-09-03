# Alterações V3 — OCR de texto fixo

Correção criada a partir de um caso real em que o take 1 continha:

`"Vai engolir tudo ou você tem nojinho?"`

e a V2 retornava fragmentos como `di engolir tudo ob vate ...`.

## Causas encontradas

1. A V2 premiava demais candidatos com mais palavras. Um OCR comprido e cheio de ruído podia ganhar de uma leitura curta e correta.
2. Quando Português era selecionado, o OCR ainda usava `por+eng`. Isso aumentava falsos positivos em palavras portuguesas.
3. Leituras diferentes eram mescladas para tentar recuperar prefixos/sufixos. Em casos ruins isso criava uma frase "Frankenstein" juntando pedaços incompatíveis.
4. Em primeiro take estático, uma alucinação do OCR pode se repetir em todos os frames; repetição temporal sozinha não garante que o texto é real.

## Correções

- Português agora usa somente o modelo `por` do Tesseract.
- Novo score de naturalidade: símbolos soltos, tokens sem letras e lixo numérico são fortemente penalizados.
- Comprimento da frase deixou de ser o principal fator de escolha.
- Escolha final usa consenso entre leituras em vez de simplesmente escolher a mais longa.
- Removida a concatenação agressiva de leituras diferentes.
- Fragmentos fracos grudados ao grupo principal são descartados.
- Prefixo/sufixo só podem ser reparados quando outra leitura concorda em pelo menos 3 palavras consecutivas do miolo.
- Aspas duplas são balanceadas quando o OCR reconhece apenas uma das bordas.

## Teste de regressão

Foi criado um vídeo de teste a partir do frame enviado pelo usuário. A V3 recuperou:

`"Vai engolir tudo ou você tem nojinho?"`

O emoji do frame não é garantido pelo Tesseract; emojis exigem uma camada separada de reconhecimento visual caso seja necessário copiá-los exatamente.
