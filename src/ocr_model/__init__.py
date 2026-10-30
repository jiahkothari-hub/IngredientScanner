"""Our own text-line recogniser for ingredient labels (the deep-learning part of the project).

Three architectures read the same synthetic text-line images and are trained with the same CTC loss,
so the only thing that changes between them is the network:
  ann   fully connected network on a sliding pixel window
  cnn   convolutional network, one prediction per image column
  crnn  the same CNN followed by a bidirectional LSTM (reads the whole line)

  python -m src.ocr_model.render_lines --samples 24      # look at the synthetic training images
  python -m src.ocr_model.train --model crnn --steps 20000
"""
