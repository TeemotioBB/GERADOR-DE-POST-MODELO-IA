# V4 — correção da primeira palavra do OCR

Correção focada no caso em que o Tesseract reconhece corretamente o corpo da frase, mas perde a primeira letra/palavra por causa de aspas, contorno ou compressão.

Exemplo corrigido:
- leitura ruim: `A engolir tudo ou você tem nojinho?`
- leitura alternativa: `Vai engolir tudo ou você tem nojinho?`

Quando pelo menos 4 palavras seguintes coincidem, a V4 permite que uma leitura lexicalmente plausível e mais informativa corrija somente a primeira palavra. Isso evita reconstruir o restante da frase ou juntar OCRs diferentes.
