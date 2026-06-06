---
type: paper-fulltext
slug: multimodal-crop-tool
arxiv_id: null
source_path: /Users/saris.kia.adm/.paper-scholar/multimodal-crop-tool/multimodal-crop-tool.md
paper: "[[multimodal-crop-tool]]"
---
<!-- fetched from https://platform.claude.com/cookbook/multimodal-crop-tool by afk (web cookbook, not an arXiv paper) -->

# Giving Claude a crop tool for better image analysis

## Overview

Give Claude a crop tool to zoom into image regions for detailed analysis of charts, documents, and diagrams.

When Claude analyzes images, it sees the entire image at once. For detailed tasks—like reading small text, comparing similar values in a chart, or examining fine details—this can be limiting.

**The solution:** Give Claude a tool that lets it "zoom in" by cropping regions of interest.

## When is a Crop Tool Useful?

- **Charts and graphs**: Comparing bars/lines that are close in value, reading axis labels
- **Documents**: Reading small text, examining signatures or stamps
- **Technical diagrams**: Following wires/connections, reading component labels
- **Dense images**: Any image where details are small relative to the whole

## Setup

```python
%pip install -q anthropic pillow datasets
```

```python
import base64
from io import BytesIO
from anthropic import Anthropic
from datasets import load_dataset
from PIL import Image as PILImage
client = Anthropic()
MODEL = "claude-opus-4-6"
```

## Define the Crop Tool

The crop tool uses **normalized coordinates** (0-1) so Claude doesn't need to know the image dimensions:

- `(0, 0)` = top-left corner
- `(1, 1)` = bottom-right corner
- `(0.5, 0.5)` = center

```python
CROP_TOOL = {
    "name": "crop_image",
    "description": "Crop an image by specifying a bounding box.",
    "input_schema": {
        "type": "object",
        "properties": {
            "x1": {"type": "number", "minimum": 0, "maximum": 1,
                   "description": "Left edge, normalized 0-1"},
            "y1": {"type": "number", "minimum": 0, "maximum": 1,
                   "description": "Top edge, normalized 0-1"},
            "x2": {"type": "number", "minimum": 0, "maximum": 1,
                   "description": "Right edge, normalized 0-1"},
            "y2": {"type": "number", "minimum": 0, "maximum": 1,
                   "description": "Bottom edge, normalized 0-1"},
        },
        "required": ["x1", "y1", "x2", "y2"],
    },
}

def handle_crop(image, x1, y1, x2, y2):
    if not all(0 <= c <= 1 for c in [x1, y1, x2, y2]):
        return [{"type": "text", "text": "Error: Coordinates must be between 0 and 1"}]
    if x1 >= x2 or y1 >= y2:
        return [{"type": "text", "text": "Error: Invalid bounding box (need x1 < x2 and y1 < y2)"}]
    w, h = image.size
    cropped = image.crop((int(x1*w), int(y1*h), int(x2*w), int(y2*h)))
    return [
        {"type": "text", "text": f"Cropped to ({x1:.2f},{y1:.2f})-({x2:.2f},{y2:.2f}): {cropped.width}x{cropped.height}px"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": pil_to_base64(cropped)}},
    ]
```

## The Agentic Loop

Send the image to Claude with the crop tool available, and handle tool calls in a loop until Claude provides a final answer.

```python
def ask_with_crop_tool(image, question):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": f"Answer the following question about this image.\n\nThe question is: {question}\n\n"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": pil_to_base64(image)}},
        {"type": "text", "text": "\n\nUse your crop_image tool to examine specific regions including legends and axes."},
    ]}]
    while True:
        response = client.messages.create(model=MODEL, max_tokens=1024, tools=[CROP_TOOL], messages=messages)
        if response.stop_reason != "tool_use":
            return
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = handle_crop(image, **block.input)
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
        messages.append({"role": "user", "content": tool_results})
```

## Summary

The crop tool pattern is simple but powerful:

1. **Define a tool** that takes normalized bounding box coordinates
2. **Return the cropped image** as base64 in the tool result
3. **Let Claude decide** when and where to crop

This works because Claude can see the full image first, identify regions that need closer inspection, and iteratively zoom in. The Claude Agent SDK provides a cleaner way to define tools using Python decorators and handles the agentic loop automatically (via `create_sdk_mcp_server` and the `@tool` decorator, exposed as an MCP server with `allowed_tools=["Read", "mcp__crop__crop_image"]`).
