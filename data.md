# Fashion MNIST Dataset import script

Here's what the script does, section by section, and why each step matters.

## `What this code does (big picture)`

* **Downloads** the Fashion-MNIST dataset (once) into a temporary folder.
* **Parses the raw IDX files** (a simple binary format) directly with `fread`.
* **Returns training/test arrays** in the exact shapes and types that MATLAB's deep learning functions (e.g., `trainnet`) expect:

  * `XTrain`: `28×28×1×60000` of type `single` scaled to `[0,1]`
  * `YTrain`: `60000×1` categorical with class names
  * Similarly for `XTest`, `YTest` (10,000 images)

---

### 1) Download Fashion-MNIST (once)

```matlab
dataDir = fullfile(tempdir,"fashion-mnist");
if ~exist(dataDir,'dir'), mkdir(dataDir); end
base = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/";
files = ["train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz", ...
         "t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz"];
for f = files
    raw = fullfile(dataDir, erase(f,".gz"));
    if ~isfile(raw)
        gz = websave(fullfile(dataDir,f), base+f);
        gunzip(gz, dataDir);
        delete(gz);
    end
end
```

* Uses a **temp directory** (`tempdir`) so it doesn't clutter your workspace and is cross-platform (`fullfile`).
* Downloads four `.gz` files (images + labels for **train** and **t10k** (test)).
* `erase(f,".gz")` computes the unzipped target path; if that file is **already present**, the script **skips** download/unzip ("download once" behavior).
* `websave` downloads, `gunzip` extracts, then the compressed file is removed.

---

### 2) Load into MATLAB arrays ready for `trainnet`

```matlab
[XTrain,YTrain] = loadFashionMNIST(dataDir,"train");
[XTest ,YTest ] = loadFashionMNIST(dataDir,"t10k");

size(XTrain)      % 28x28x1x60000
size(XTest)       % 28x28x1x10000
categories(YTrain)% 10 class names
```

* Calls a helper to read and convert binary IDX files to the canonical **4-D image array** layout MATLAB uses: `[H W C N]`.
* Labels are returned as a **categorical** vector with human-readable class names.

---

### 3) Helper: `loadFashionMNIST`

```matlab
function [X,Y] = loadFashionMNIST(dataDir, split)
    switch split
        case "train"
            img = fullfile(dataDir,"train-images-idx3-ubyte");
            lbl = fullfile(dataDir,"train-labels-idx1-ubyte");
        case "t10k"
            img = fullfile(dataDir,"t10k-images-idx3-ubyte");
            lbl = fullfile(dataDir,"t10k-labels-idx1-ubyte");
        otherwise
            error("split must be ""train"" or ""t10k"".");
    end
    X = readIDXImages(img);      % 28x28x1xN single in [0,1]
    Y = readIDXLabels(lbl);      % Nx1 categorical
end
```

* Picks the correct pair of files for the chosen split.
* Delegates to the two readers that know the IDX format.

---

### 4) Helper: `readIDXImages` (parses IDX image file)

```matlab
function X = readIDXImages(filename)
    fid = fopen(filename,'r','ieee-be');
    magic     = fread(fid,1,'uint32'); assert(magic==2051,"Bad image magic");
    numImages = fread(fid,1,'uint32');
    numRows   = fread(fid,1,'uint32');
    numCols   = fread(fid,1,'uint32');
    raw = fread(fid, numImages*numRows*numCols, 'uint8=>single'); fclose(fid);
    raw = raw/255;
    X = reshape(raw, [numRows,numCols,numImages]);  % row-major → fix orientation
    X = permute(X,[2 1 3]);
    X = reshape(X, [numCols,numRows,1,numImages]);  % 28x28x1xN
end
```

* **`fopen(...,'ieee-be')`**: The IDX header is **big-endian**; this ensures the 32-bit integers are read correctly.
* **Magic number check**: `2051` (hex `0x00000803`) identifies an **image** file with `0x08` = unsigned byte data and `0x03` = 3 dimensions (rows, cols, items). The assert guards against wrong/corrupt files.
* Reads counts: number of images, rows, columns.
* **Pixel read**: `uint8=>single` converts to `single` during read; then `/255` scales to `[0,1]`.
* **Reshape & permute**:

  * IDX stores data in **row-major** order (C-style). MATLAB is **column-major** (Fortran-style).
  * First `reshape` to `[numRows,numCols,numImages]`, then `permute(X,[2 1 3])` **swaps rows/cols** to fix orientation.
  * Final `reshape` adds the **channel dimension** (grayscale ⇒ `1`) and places images in the 4th dim, yielding `[H W C N]` = `[28 28 1 N]`.

---

### 5) Helper: `readIDXLabels` (parses IDX label file)

```matlab
function Y = readIDXLabels(filename)
    fid = fopen(filename,'r','ieee-be');
    magic     = fread(fid,1,'uint32'); assert(magic==2049,"Bad label magic");
    numItems  = fread(fid,1,'uint32');
    labels    = fread(fid, numItems, 'uint8=>double'); fclose(fid);
    names = ["T-shirt/top","Trouser","Pullover","Dress","Coat", ...
             "Sandal","Shirt","Sneaker","Bag","Ankle boot"];
    Y = categorical(labels,0:9,names);
end
```

* **Magic number** `2049` (hex `0x00000801`) identifies a **label** file (1 dimension of items).
* Reads `numItems` then the labels as bytes.
* Converts numeric labels `0…9` into a **categorical** vector with the canonical Fashion-MNIST class names. This is handy for:

  * readable metrics and charts,
  * ensuring class order is fixed and known to the training code.

---

### Outputs & why the shapes/types matter

* `X*` is `single`, **scaled** to `[0,1]`, and shaped `[Height Width Channels NumObservations]`. This matches what layers and functions like `imageInputLayer` / `trainNetwork` / `trainnet` expect.
* `Y*` is a **categorical** vector (`N×1`), which is what classifiers (`classificationLayer`, cross-entropy losses) expect.

---

### Small notes / gotchas

* If you ever see flipped images, the culprit is usually the **row/column** interpretation; this code fixes it via `permute`.
* To save RAM you could keep the raw as `uint8` and scale later in a datastore/mini-batch pipeline, but for Fashion-MNIST the full in-memory `single` is typically fine.
* The **"t10k"** split is the standard 10k test set naming from MNIST/Fashion-MNIST.

That's it—clean, minimal IDX parsing wired to MATLAB's canonical image/label formats so you can train immediately.

## `What is an IDX file?`

An **IDX file** is a tiny, binary container format used by the (Fashion-)MNIST datasets to store arrays (vectors, matrices, tensors). It's very simple: a short header that describes the data type and shape, followed by the raw values in row-major order (C-style), big-endian.

### Structure at a glance

* **Endian:** big-endian (be careful—MATLAB defaults to little-endian unless you specify).
* **Header (first bytes):**

  * Byte 0: `0x00`
  * Byte 1: `0x00`
  * Byte 2: **data type code**
  * Byte 3: **number of dimensions** (ndim)
  * Next `ndim` × 4 bytes: **dimension sizes** as 32-bit big-endian integers (e.g., N, rows, cols)
* **Data block:** raw values for the array, contiguous, **row-major** (so in MATLAB you typically swap row/col or `permute` after reading).

### Common data type codes

* `0x08` → unsigned byte (1 byte)  ← used by MNIST images/labels
* `0x09` → signed byte
* `0x0B` → short (2 bytes)
* `0x0C` → int (4 bytes)
* `0x0D` → float (4 bytes)
* `0x0E` → double (8 bytes)

### How this maps to MNIST/Fashion-MNIST

* **Images file:** magic bytes `00 00 08 03` → `dtype=ubyte`, `ndim=3`; dims are `[numImages, nRows, nCols]`.
  When read as a big-endian uint32 "magic number," this appears as **2051**.
* **Labels file:** magic bytes `00 00 08 01` → `dtype=ubyte`, `ndim=1`; dims are `[numItems]`.
  As uint32 big-endian, this appears as **2049**.

### Why your MATLAB code does the swaps

* You read with `'ieee-be'` to match **big-endian**.
* You **reshape** according to dims and then **permute rows/cols** because IDX stores data **row-major**, while MATLAB uses **column-major** and expects images in `[H W C N]`.

## `what is big endian?`

Big-endian is a **byte order** for storing multi-byte numbers in memory or files.

* **Big-endian:** the **most significant byte (MSB)** comes first (at the lowest memory address / written first on the wire).
* **Little-endian:** the **least significant byte (LSB)** comes first.

### Quick example (32-bit integer)

Number `0x12 34 56 78` stored as bytes:

* **Big-endian:** `12 34 56 78`
* **Little-endian:** `78 56 34 12`

If you read bytes in the wrong endianness, you get the wrong value.
The IDX image "magic" bytes are `00 00 08 03`:

* Interpreted **big-endian** → `2051` (correct)
* Interpreted **little-endian** → `50,855,936` (nonsense for this header)

### Why you care

* Endianness matters only for **multi-byte binary numbers** (ints/floats). Text and single bytes aren't affected.
* Network protocols traditionally use **big-endian** ("network byte order").
* Most desktop CPUs (x86/x64) are **little-endian**, so software must **explicitly** specify endianness when reading/writing binary formats.

### In practice

* **MATLAB:** `fopen(file,'r','ieee-be')` reads big-endian; use `'ieee-le'` for little-endian.
* **Python (struct):** `struct.unpack('>I', b)` for big-endian (`>`), `<I` for little-endian.
* **Rule of thumb:** follow the file/protocol spec. For IDX (MNIST/Fashion-MNIST), headers and sizes are **big-endian**, so your code uses `'ieee-be'` correctly.

> Side note: **Endianness ≠ bit order.** Bit significance inside a byte is a different concept; most formats use the same bit order regardless of byte order.


## Resources

- [Endianness Explained in Less Than 9 Minutes](https://www.youtube.com/watch?v=n6lPuep66aM&ab_channel=KeaSigmaDelta)