# Segment Anything

## Situation for Geti version 3.1

### SAM v1 in the UI

The web UI ships **Segment Anything Model version 1 (SAM v1)** to power the
interactive **"Segment Anything"** annotation tool. SAM v1 runs entirely
client-side in the browser, so a user can hover/click on an object and get an
instant object mask without a round-trip to the backend.

#### How it runs

- The model is executed inside a single shared web worker
  (`src/features/annotator/webworkers/segment-anything.worker`), which hosts
  **both** the SAM v1 encoder and decoder ONNX sessions
  (`SEGMENT_ANYTHING_ENCODER` and `SEGMENT_ANYTHING_DECODER`) via ONNX Runtime
  Web (WASM/JSEP). Keeping both sessions in one worker avoids doubling the
  OpenCV + ONNX Runtime footprint, since the encoder and decoder always run
  sequentially.
- Inference is a two-step pipeline:
  1. **Encoding** - the current image is encoded once
     (`processEncoder`) and the resulting embedding is cached (encodings for the
     next media item are also prefetched to make navigation feel instant).
  2. **Decoding** - user prompt points are fed to the decoder
     (`processDecoder`) against the cached encoding to produce a mask. The mask
     is post-processed to a `polygon` (or a bounding `rect` for detection tasks).
     Hovering decodes a single point under the cursor. **Shift + click**
     (include) and **Alt/Option + click** (exclude) place prompt points instead;
     each added point passes the previous low resolution mask back to the
     decoder (`previousMask`) to refine the same object. A plain click creates
     the annotation and **Esc** clears the points.
- **Mask-to-polygon post-processing** relies on the custom-compiled
  `opencv.js` bundled by `@geti-ui/smart-tools`.

#### Disadvantages of the current approach

The current SAM v1 in-browser setup has a number of drawbacks that together
degrade the user experience:

- **Outdated model version.** SAM v1 is now several generations behind newer
  segmentation models (e.g. SAM 2 and other successors). It produces
  lower-quality masks on thin structures, small objects, and cluttered scenes,
  and lacks improvements in prompt handling and boundary accuracy that later
  versions provide. Staying on v1 means users get worse suggestions than the
  current state of the art.
- **Unstable encoder execution.** Running the encoder ONNX session in the
  browser via ONNX Runtime Web (WASM/JSEP) is fragile: it is sensitive to the
  browser, hardware, and available memory, and can fail, stall, or return
  inconsistent embeddings. Because the whole pipeline depends on the cached
  encoding, an unstable encoder step makes the tool unreliable and hard to
  reason about when something goes wrong.
- **No WebGPU acceleration in the embedded WebView.** The core performance
  bottleneck is that ONNX Runtime Web's WebGPU/JSEP path requires
  `SharedArrayBuffer` and cross-origin isolation, which the embedded WebView
  used by the Tauri app does not provide. As a result, our more recent updates
  to the SAM logic had to drop WebGPU support under Tauri and fall back
  immediately to CPU execution, leaving the user waiting noticeably longer for
  a SAM result.
- **Sub-optimal performance.** With WebGPU unavailable, the encoder and decoder
  run on the CPU client-side within the constraints of the Tauri app, so
  inference is computationally heavy and incurs high latency. Encoding is slow,
  consumes significant memory, and competes with the rest of the UI for
  resources, causing noticeable jank and occasional freezes — especially on
  lower-end machines and with large images.
- **Bad user experience.** The combination of dated mask quality, an unstable
  encoder, and sluggish in-browser inference leads to slow, inconsistent, and
  sometimes broken interactions.


## Geti 3.2 improvements

### Choosing a SAM model for the new approach

To move past the limitations of the SAM v1 in-browser setup, we evaluated the
available Segment Anything variants and selected the one that best balances
license, accuracy, and speed:

- **SAM 3** - Despite being the newest option, its
  license terms are incompatible with our distribution and usage requirements,
  so it cannot be adopted for Geti. It introduces the ability to use open-vocabulary prompts, 
  but as a drawback SAM 3 models have larger memory requirements and slower inference
- **SAM 1** - This is the model we are moving away from. It is several
  generations behind, with lower mask quality and weaker prompt handling than
  current models.
- **Branches, forks, and optimized versions - not state of the art.** Community
  forks and performance-optimized derivatives can offer niche speed or size
  benefits, but they are not state of the art in accuracy or robustness, and
  they add maintenance and support risk without a clear quality upside.
- **SAM 2** - This model is compatible with licensing requirements
  and is both **faster** and **more accurate** than SAM v1. It delivers
  state-of-the-art segmentation quality making it the right foundation for the 
  new approach. This model(-s) can also be easily converted to OpenVino IR format for 
  Intel GPU optimized inference.

### Where to run encoding and decoding

With SAM 2 selected, the next decision is how to split the inference pipeline
between the backend and the client. Two architectural approaches are available taking 
into account issues running both encoder and decoder in browser.

#### Option 1: Encoder and decoder on the backend

The backend runs both the encoder and the decoder and streams the resulting
predictions (masks/polygons) to the client.

**Pros**

- **Consistent, powerful compute.** Inference runs on server hardware (optionally
  GPU-accelerated), giving predictable performance regardless of the user's
  device or browser.
- **Simpler, lighter client.** No ONNX Runtime, WASM, or OpenCV bundle shipped to
  the browser, reducing download size and memory pressure.
- **Easier maintenance and updates.** The model lives in one place, so upgrading
  weights, tuning post-processing, or applying fixes does not require shipping a
  new frontend build.
- **Better security and IP control.** Model weights never leave the server.

**Cons**

- **A round-trip per interaction.** Every prompt (each click/box) requires a
  request to the backend, so interactive latency is bound by network round-trip
  time. This can feel sluggish and undermines the "instant mask" experience.
- **Higher backend load and cost.** Decoding is invoked frequently during
  annotation; running every prompt server-side multiplies request volume and
  compute cost, and requires scaling for concurrent users.
- **Weaker offline / poor-network behavior.** The tool is unusable or degraded
  when connectivity is slow or interrupted.

#### Option 2: Encoder on the backend, decoding in the browser

The backend runs the (heavy) encoder once per image and sends the image
embedding to the client. The client caches the embedding and runs the
lightweight decoder locally for each prompt.

**Pros**

- **Fast, interactive decoding.** Once the embedding is available, each prompt is
  decoded locally with no network round-trip, so masks update in near real time
  as the user clicks — the responsiveness that makes the tool feel instant.
- **Lower backend load per interaction.** The expensive encoding runs only once
  per image (and can be prefetched), while frequent decoding is offloaded to the
  client, reducing server request volume and cost.
- **Resilient to network hiccups.** After the embedding is fetched, additional
  prompts keep working even if the connection is briefly degraded.

**Cons**

- **Heavier, more complex client.** The browser must ship and run the decoder via
  ONNX Runtime Web plus OpenCV post-processing, increasing bundle size, memory
  use, and the risk of device/browser-specific issues.
- **Embedding transfer overhead.** Image embeddings are relatively large; sending
  them to the client adds bandwidth cost and an initial latency before the first
  prompt can be decoded.
- **Split pipeline to maintain.** Logic lives on both sides (backend encoder,
  client decoder), so versions must stay compatible and changes must be
  coordinated across backend and frontend.
- **Some model exposure.** The decoder and embeddings are delivered to the
  client, offering less IP protection than a fully server-side approach.

Weighing these trade-offs, the superior user experience of second option outweighs
its cons.

### SAM 2.1 Models Description

| Model                  | Size (M) | Speed (FPS)   |  SA-V test (J&F) | MOSE val (J&F) | LVOS v2 (J&F) |
|------------------------|---------:|--------------:|-----------------:|---------------:|--------------:|
| sam2.1_hiera_tiny      |     38.9 |          91.2 |             76.5 |           71.8 |          77.3 |
| sam2.1_hiera_small     |     46   |          84.8 |             76.6 |           73.5 |          78.3 |
| sam2.1_hiera_base_plus |     80.8 |          64.1 |             78.2 |           73.7 |          78.2 |
| sam2.1_hiera_large     |    224.4 |          39.5 |             79.5 |           74.6 |          80.6 |

### SAM 2.1 models benchmarks

A few quick benchmarks were run to compare the performance of the SAM 2.1 models on both GPU and CPU. The results are summarized below.
All models were converted to OpenVino IR format and run with OpenVino Runtime. The following metrics were collected:

- EncCold = time to load encoder model without OpenVino cache enabled (cold start)
- EncWarm = time to load encoder model with OpenVino cache enabled
- PredCold = time to load decoder model without OpenVino cache enabled (cold start)
- PredWarm = time to load decoder model with OpenVino cache enabled
- Enc mean = average time to run encoder model
- Enc fps = frames per second for encoder model
- Pred mean = average time to run decoder model
- Pred fps = frames per second for decoder model
- E2E mean = average time to run encoder and decoder models end-to-end (excluding model loading)
- E2E fps = frames per second for encoder and decoder models end-to-end (excluding model loading)

Load/latency in ms; Cold = first compile, Warm = from cache.

#### GPU (Panther Lake)

| Model     | EncCold | EncWarm | PredCold | PredWarm | Enc mean | Enc fps | Pred mean | Pred fps | E2E mean | E2E fps |
| --------- | ------: | ------: | -------: | -------: | -------: | ------: | --------: | -------: | -------: | ------: |
| tiny      |  4208.7 |   147.8 |    282.8 |     50.8 |    88.66 |   11.28 |     10.93 |    91.53 |    99.59 |   10.04 |
| small     |  4133.9 |   121.9 |    280.9 |     51.4 |   102.20 |    9.79 |     10.92 |    91.57 |   113.12 |    8.84 |
| base_plus |  5195.9 |   221.9 |    297.1 |     53.0 |   164.27 |    6.09 |     10.95 |    91.31 |   175.22 |    5.71 |
| large     |  6826.8 |   413.0 |    290.6 |     55.9 |   364.58 |    2.74 |     11.21 |    89.23 |   375.78 |    2.66 |

#### CPU (Tiger Lake)

| Model     | EncCold | EncWarm | PredCold | PredWarm | Enc mean | Enc fps | Pred mean | Pred fps | E2E mean | E2E fps |
| --------- | ------: | ------: | -------: | -------: | -------: | ------: | --------: | -------: | -------: | ------: |
| tiny      |  7682.0 |   292.7 |    320.5 |     97.7 |  2002.11 |    0.50 |     97.28 |    10.28 |  2099.39 |    0.48 |
| small     |  8187.2 |   383.7 |    332.2 |    107.4 |  2555.28 |    0.39 |     74.38 |    13.44 |  2629.65 |    0.38 |
| base_plus | 10728.1 |   616.8 |    364.6 |    101.7 |  4210.63 |    0.24 |     65.58 |    15.25 |  4276.21 |    0.23 |
| large     | 14116.6 |  1472.8 |    386.9 |    104.3 | 10906.51 |    0.09 |     59.97 |    16.67 | 10966.48 |    0.09 |

### MobileSAM 1 encoding benchmarks (baseline)

The same tests were done on the original MobileSAM 1 TinyViT encoder model to provide a baseline for comparison. 
The results are summarized below.

#### GPU (Panther Lake)

| EncCold | EncWarm | Enc mean | Enc fps |
| ------: | ------: | -------: | ------: |
|   520.3 |   171.5 |    51.47 |   19.43 |

#### CPU (Tiger Lake)

| EncCold | EncWarm | Enc mean | Enc fps |
| ------: | ------: | -------: | ------: |
|   569.0 |   179.5 |   439.26 |    2.28 |

#### Choosing a model for CPU targets

The Segment Anything tool is intended to run on CPU-only machines as well, not
just GPU-equipped ones, so we base our model choice on the worst-case (CPU)
scenario.

- **`large` and `base_plus` are ruled out.** On CPU their end-to-end throughput
  is far too low (0.09 and 0.23 E2E fps respectively), which would make the
  encoding step feel unacceptably slow for interactive annotation.
- **Choose between `tiny` and `small`.** These are the only models with a
  workable CPU profile, so the decision comes down to these two.
- **Proceed with `small`, fall back to `tiny` if needed.** Starting with `small`
  for its better mask quality, and if the user experience proves insufficient on
  CPU it can be switched to `tiny` for faster encoding.
- **Decoding latency is good for both.** Importantly, both `tiny` and `small`
  provide good decoding latency (~74-97 ms / 10-13 fps on CPU), so the
  interactive, client-side decoding step remains responsive regardless of which
  of the two we pick.

#### Eliminating cold-start model loading

The benchmarks show a large gap between cold and warm load times (e.g. several
seconds on the first compile versus a fraction of a second from cache). To avoid
exposing this cold-start delay to the user, the model can be **pre-loaded during
application startup and cached**. Warming the encoder ahead of time
(and reusing the OpenVino cache) means that by the time a user reaches for the
Segment Anything tool the model is already compiled and ready, so the first
interaction is fast instead of incurring the one-time cold-load penalty.

### Image embedding endpoint

_Endpoint:_ `GET /api/projects/{project_id}/dataset/media/{media_id}/embeddings`

_Response:_ HTTP 200 OK and image embedding in `safetensors` format (`Content-Type: application/octet-stream`) 

### Implementation plan

Phase 1: Keeping original MobileSAM 1 in the UI, but moving encoding to the backend
1. The new endpoint is introduced to the backend API, which runs the encoder and returns the embedding to the client
2. Package `@geti-ui/smart-tools` is adjusted to be able to parse server-side generated embeddings and feed them to the decoder for mask generation
3. The UI application is switched to new `@geti-ui/smart-tools` and adjusted to obtain embeddings from the backend instead of running the encoder in the browser

Phase 2: Upgrading to SAM 2.1
4. SAM 2 support is added to the `model_api` module. 
   New classes inheriting `ImageModel` class from `model_api` should be added for both encoder and decoder similarly 
   to SAM 1 support: https://github.com/open-edge-platform/model_api/blob/master/model_api/src/model_api/models/sam_models.py
   Benefit of adding it to `model_api` is that it can be reused later in other products, plus it already has built-in code and algorithma
   for data pre- and post-processing.
5. Encoder model weights are converted to OpenVino IR format and added to the backend replacing old MobileSAM 1 encoder. The backend endpoint is adjusted to use the new encoder model.
6. Decoder model weights are downloaded and converted to ONNX format and added to `@geti-ui/smart-tools` for client-side decoding.

### Open points

1. Choosing between `sam2.1_hiera_tiny` and `sam2.1_hiera_small` for the default model. The decision will be based on user experience testing, balancing mask quality against encoding speed on CPU.
2. By default, encoding will be done on CPU not to interfere with GPU resources for other tasks. However, if the user has a GPU available, we may consider adding an option to run encoding on GPU for faster performance.
3. By default, model is automatically unloaded after encoding to free up memory. However, we may consider adding an option to keep the model loaded with certain timeout\threshold for faster subsequent encodings if the user is working with multiple images in a session.




