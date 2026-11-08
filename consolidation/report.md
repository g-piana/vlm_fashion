# Taxonomy reconciliation — client `base` (fingerprint 2fb9b7ed1bbc)

**129 values to decide** in `decisions.yaml`.


## Attribute values

| App field | VLM values | Match | To decide | App-only |
|---|---|---|---|---|
| colours | 47 | 47 | 0 | 0 |
| embellishments | 41 | 41 | 0 | 1 |
| finishing | 22 | 22 | 0 | 17 |
| material | 53 | 53 | 0 | 0 |
| mod_details[accessories] | 6 | 0 | 6 | 0 |
| mod_details[bags] | 32 | 26 | 6 | 16 |
| mod_details[luggage] | 4 | 0 | 4 | 13 |
| mod_details[ready-to-wear] | 73 | 72 | 1 | 1 |
| mod_details[shoes] | 28 | 28 | 0 | 8 |
| pattern | 25 | 25 | 0 | 0 |

## Categories and subcategories

- `handbags-women` app-only subcategories (VLM never predicts): handbag
- `clutches-women` app-only subcategories (VLM never predicts): minaudiére, pochette
- `shoulder-bags-women` app-only subcategories (VLM never predicts): chain bag, crossbody bag, shoulder bag
- `flat-shoes-men` app-only subcategories (VLM never predicts): mule, slipper
- `sandals-men` app-only subcategories (VLM never predicts): sandal
- `flat-shoes-women` app-only subcategories (VLM never predicts): lace up, slipper
- `heeled-shoes-women` app-only subcategories (VLM never predicts): lace up, mule
- `sandals-women` app-only subcategories (VLM never predicts): mule, sandal
- `boots-and-booties-women` app-only subcategories (VLM never predicts): shoe boot

## VLM attributes without an app field

Add `app: {field: <import field>}` or `app: {field: null}` to each:

- accessories.yaml: accessory_closure
- eyewear.yaml: frame_shape, lens_color, frame_material
- jewelry.yaml: jewelry_material, gemstone
- luggage.yaml: luggage_type
- ready_to_wear.yaml: fit, length, neckline, collar, sleeve_length, sleeve_type, closure, waist, lining
- watches.yaml: dial_shape, strap_type, movement
