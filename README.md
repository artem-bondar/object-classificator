# Object classificator

A simple object classificator created under CMC MSU computer graphics course.
For task requirements see *PRD.pdf*.

## Build

In project directory run:

```bash
make all
```

Then to train classifier model from `build/bin` run:

```bash
./task2 -d ../../../data/train_labels.txt -m model.txt --train
```

To classify images from test sample using trained classifier run:

```bash
./task2 -d ../../../data/test_labels.txt -m model.txt -l predictions.txt --predict
```

To compare accuracy of classification run from `/project`:

```bash
./compare.py ../data/test_labels.txt build/bin/predictions.txt
```

## Screenshots

![Konsole](docs/images/Screenshot&#32;1.png)
