# Revisão técnica — Mídia + Vídeo Automático

## O que o sistema faz hoje

1. Importa um Reel público do Instagram (ou recebe um arquivo local).
2. Analisa onde termina o primeiro take.
3. Obtém a legenda do primeiro take por OCR, Whisper ou texto manual.
4. Substitui o primeiro take por uma foto/vídeo da personagem.
5. Renderiza a legenda revisada sobre a nova mídia.
6. Mantém o áudio original e volta ao vídeo original após a transição.
7. Entrega MP4 H.264 pronto para uso.

## Problemas encontrados e corrigidos

### 1. OCR juntava texto aleatório da tela
**Antes:** cada frame era tratado quase como um bloco único (`PSM 6`), e a pontuação favorecia quantidade de palavras. Placas, logos, watermark e texto de cenário podiam entrar na frase.

**Agora:** o OCR mantém posição dos blocos, usa `PSM 11`, compara vários frames e só fortalece texto que reaparece no mesmo lugar. Textos de cenário que mudam de posição/conteúdo perdem para uma legenda estável.

### 2. “Mais palavras” podia ganhar mesmo estando errado
**Antes:** cada palavra adicional acrescentava pontuação sem um teto relevante.

**Agora:** completude tem limite e a persistência temporal vale muito mais. Um texto comprido e aleatório não vence apenas por ser comprido.

### 3. Português misturava reconhecimento em inglês
**Antes:** mesmo escolhendo Português, o OCR podia usar `por+eng`.

**Agora:** Português usa `por`. `por+eng` fica reservado para o modo automático.

### 4. O caractere `|` virava `I`
Essa substituição podia transformar ruído visual em letra e contribuir para resultados inventados. Foi removida.

### 5. OCR fraco ainda preenchia o campo
**Agora:** se o texto não reaparecer com evidência suficiente, o sistema devolve campo vazio. Para produção em escala, falso negativo é melhor do que publicar uma legenda errada.

### 6. Sem controle de área do OCR
Foi adicionado o seletor **Onde procurar o texto escrito**:
- Automática (recomendado)
- Parte superior
- Centro
- Parte inferior
- Tela quase inteira

Isso resolve rapidamente padrões em que a legenda sempre fica em uma faixa específica.

### 7. A resolução final herdava a mídia da personagem
Uma foto 4:5 ou quadrada podia mudar o canvas de um Reel 9:16. Agora a saída preserva as dimensões/proporção do vídeo original e encaixa a nova mídia nesse canvas.

### 8. Continuação de vídeo tinha ~70 ms de deslocamento
O vídeo pulava 0,07 s após a transição, mas o áudio não. Isso criava um pequeno descompasso A/V. Agora os dois usam exatamente o mesmo timestamp.

### 9. Código de importação duplicado
`url_import.py` repetia praticamente todo o código de `core/instagram_import.py`. Agora ele é apenas uma camada de compatibilidade, deixando uma única implementação real.

### 10. Rota externa podia gerar custo sem autenticação
A interface Gradio podia ter senha, mas `/api/import-video` era uma rota FastAPI separada. Agora ela fica desativada se `API_TOKEN` estiver vazio. Para integração externa, configure um token e envie-o no header `X-API-Token`.

## Validações realizadas

- Compilação de todos os módulos Python: OK.
- OCR sintético com legenda principal + watermark persistente + texto de cenário mutável: legenda principal selecionada corretamente.
- OCR com legenda curta e watermark de três palavras: legenda central selecionada corretamente.
- Render com fonte 9:16 e mídia inicial quadrada: saída preservou 9:16.
- Render com áudio, 30 FPS e transição manual: vídeo e áudio finalizaram com 3,00 s e 30 FPS.

## Limitação importante do produto atual

O modo **Copiar texto escrito no vídeo** foi desenhado para uma **legenda fixa no primeiro take**. Se o texto muda palavra por palavra, troca de frase durante o take ou é legenda estilo karaoke, o produto precisa de um segundo modo de OCR temporal. Não é recomendável tentar “forçar” o modo fixo a resolver isso, porque ele voltaria a misturar frases.

## Próximos upgrades que mais valem a pena

### Prioridade 1 — Fila de revisão por confiança
Em lote, vídeos com OCR forte podem seguir automaticamente e apenas os de baixa confiança vão para revisão manual. Isso reduz muito o trabalho humano sem publicar texto errado.

### Prioridade 2 — Diagnóstico visual do OCR
Salvar uma imagem de diagnóstico mostrando o retângulo que foi escolhido como legenda. Se der erro, fica óbvio se o sistema mirou em watermark, placa ou texto principal.

### Prioridade 3 — Modo “texto dinâmico”
OCR temporal com eventos por intervalo para vídeos em que a legenda muda. Deve ser separado do modo de frase fixa.

### Prioridade 4 — QA automático da transição
Gerar miniaturas de 0,15 s antes/depois da troca e detectar se a personagem original ainda apareceu no primeiro trecho.

### Prioridade 5 — Operação em lote
Tabela por job com: URL, status, transição, texto detectado, confiança, revisão necessária, resultado e erro. Depois disso, geração em lote fica administrável.

### Prioridade 6 — Biblioteca da modelo
Selecionar automaticamente entre vários vídeos/fotos de entrada da mesma personagem para diminuir repetição visual, mantendo proporção 9:16.

### Prioridade 7 — Observabilidade
Registrar por job as leituras candidatas do OCR, região, confiança, número de frames em que apareceu e motivo de rejeição. Isso torna ajustes futuros objetivos em vez de tentativa e erro.
