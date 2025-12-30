# Swap a single source face into a specific target face in a target image containing multiple faces

## Request

`target_index` is used to specify the index of the face that
should be replaced in a target image that has multiple faces,
for example `0` would be the first face, `1` would be the second
face, and so on.  In the example payload below, the second
face will be swapped in the target image.

**Note:** Both `source_image` and `target_image` support either base64 encoded image data OR a URL to an image (http:// or https://).

### Using Base64

```json
{
  "input": {
    "source_image": "base64 encoded source image content",
    "target_image": "base64 encoded target image content",
    "source_indexes": "-1",
    "target_indexes": "1",
    "background_enhance": true,
    "face_restore": true,
    "face_upsample": true,
    "upscale": 1,
    "codeformer_fidelity": 0.5,
    "output_format": "JPEG"
  }
}
```

### Using URLs

```json
{
  "input": {
    "source_image": "https://example.com/source-face.jpg",
    "target_image": "https://example.com/target-image.jpg",
    "source_indexes": "-1",
    "target_indexes": "1",
    "background_enhance": true,
    "face_restore": true,
    "face_upsample": true,
    "upscale": 1,
    "codeformer_fidelity": 0.5,
    "output_format": "JPEG"
  }
}
```

## Response

## RUN

```json
{
  "id": "83bbc301-5dcd-4236-9293-a65cdd681858",
  "status": "IN_QUEUE"
}
```

## RUNSYNC

### Response with R2 configured (returns URL)

```json
{
  "delayTime": 20275,
  "executionTime": 43997,
  "id": "sync-a3b54383-e671-4e24-a7bd-c5fec16fda3b",
  "output": {
    "image_url": "https://your-cdn.com/faceswap/job-id.jpg"
  },
  "status": "COMPLETED"
}
```

### Response without R2 (returns base64)

```json
{
  "delayTime": 20275,
  "executionTime": 43997,
  "id": "sync-a3b54383-e671-4e24-a7bd-c5fec16fda3b",
  "output": {
    "image": "base64 encoded output image"
  },
  "status": "COMPLETED"
}
```
