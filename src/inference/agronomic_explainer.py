"""Agronomic feature translation and domain-specific interpretation engine.

Translates technical vision descriptors (e.g. color_lab_b_mean, shape_eccentricity,
texture_glcm_contrast) into agronomic terminology understandable by agricultural
scientists, grain inspectors, millers, and farmers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


FEATURE_AGRONOMIC_MAP: Dict[str, Dict[str, str]] = {
    # Shape & Morphology
    "shape_area": {
        "term": "Kernel Surface Area",
        "category": "Morphology",
        "description": "Total projected grain area; indicates kernel plumpness and development.",
    },
    "shape_perimeter": {
        "term": "Kernel Boundary Perimeter",
        "category": "Morphology",
        "description": "Outer perimeter contour of the kernel.",
    },
    "shape_major_axis_length": {
        "term": "Grain Length (Longitudinal Axis)",
        "category": "Morphology",
        "description": "Longest dimension along the kernel; crucial for rice grain length grading.",
    },
    "shape_minor_axis_length": {
        "term": "Grain Breadth / Transverse Thickness",
        "category": "Morphology",
        "description": "Kernel width across the widest cross-section.",
    },
    "shape_aspect_ratio": {
        "term": "Length-to-Width Ratio (L/W)",
        "category": "Morphology",
        "description": "Standard international index classifying rice as slender, medium, or bold.",
    },
    "shape_eccentricity": {
        "term": "Grain Slenderness / Elongation",
        "category": "Morphology",
        "description": "Degree of elliptical stretching; broken grains show markedly lower eccentricity.",
    },
    "shape_solidity": {
        "term": "Contour Solidity & Compactness",
        "category": "Morphology",
        "description": "Measures contour indentation; fissures, bites, or seed damage lower this value.",
    },
    "shape_extent": {
        "term": "Bounding Box Occupancy",
        "category": "Morphology",
        "description": "Fraction of the grain's bounding rectangle filled by the kernel.",
    },
    "shape_height": {
        "term": "Vertical Kernel Dimension",
        "category": "Morphology",
        "description": "Projected height of standardized grain envelope.",
    },
    "shape_width": {
        "term": "Horizontal Kernel Dimension",
        "category": "Morphology",
        "description": "Projected width of standardized grain envelope.",
    },

    # Color & Chromaticity
    "color_lab_l_mean": {
        "term": "Lightness & Translucency (CIE L*)",
        "category": "Color",
        "description": "Milling degree and chalkiness; immature or chalky grains have high opaque L* values.",
    },
    "color_lab_a_mean": {
        "term": "Red-Green Chromatic Shift (CIE a*)",
        "category": "Color",
        "description": "Reddish rot or microbial discoloration index.",
    },
    "color_lab_b_mean": {
        "term": "Yellow-Blue Fungal/Heat Index (CIE b*)",
        "category": "Color",
        "description": "Key marker for mold, aflatoxin, fusarium staining, and stack-burn yellowing.",
    },
    "color_hsv_h_mean": {
        "term": "Dominant Color Hue Angle",
        "category": "Color",
        "description": "Spectral tone of the pericarp and aleurone layers.",
    },
    "color_hsv_s_mean": {
        "term": "Discoloration Saturation Intensity",
        "category": "Color",
        "description": "Vividness of surface lesions, stains, and fungal mycelia.",
    },
    "color_hsv_v_mean": {
        "term": "Surface Brightness (Value)",
        "category": "Color",
        "description": "Overall reflective illumination of the kernel coat.",
    },
    "color_rgb_r_mean": {
        "term": "Red Channel Reflectance",
        "category": "Color",
        "description": "Reflectance in red spectrum; shifts with heat damage and storage browning.",
    },

    # Texture & Surface
    "texture_glcm_contrast": {
        "term": "Surface Fissuring & Roughness",
        "category": "Texture",
        "description": "High contrast signals surface cracks, stress fissures, or fungal lesion margins.",
    },
    "texture_glcm_dissimilarity": {
        "term": "Local Texture Variation",
        "category": "Texture",
        "description": "Spatial variance across neighbouring pixels; uneven grains indicate disease.",
    },
    "texture_glcm_homogeneity": {
        "term": "Pericarp Surface Uniformity",
        "category": "Texture",
        "description": "Smoothness of the outer polished grain surface.",
    },
    "texture_glcm_energy": {
        "term": "Textural Orderliness / Regularity",
        "category": "Texture",
        "description": "Uniformity of gray-level transitions across the kernel.",
    },
    "texture_glcm_correlation": {
        "term": "Linear Texture Pattern Continuity",
        "category": "Texture",
        "description": "Directional structural alignment of the grain's cellular texture.",
    },
}

CLASS_AGRONOMIC_PROFILES: Dict[str, Dict[str, str]] = {
    "0_NOR": {
        "name": "Normal Sound Kernel",
        "grade_impact": "Grade-A Premium Stock",
        "cause": "Optimal growth, uniform maturation, and proper post-harvest drying.",
        "action": "Maintain grain moisture below 14% to preserve premium milling yield.",
    },
    "1_F&S": {
        "name": "Fusarium and Spot Blight",
        "grade_impact": "Infectious Fungal Defect",
        "cause": "Infection by Fusarium moniliforme or Bipolaris oryzae during heading or humid storage.",
        "action": "Quarantine affected lot; lower storage relative humidity below 65%. Rescreen seed.",
    },
    "2_SD": {
        "name": "Seed / Insect Damage",
        "grade_impact": "Physical & Pest Defect",
        "cause": "Pest predation (rice weevil Sitophilus oryzae) or mechanical injury during threshing.",
        "action": "Aspirate and color-sort to eliminate insect-bitten kernels. Check fumigation schedule.",
    },
    "3_MY": {
        "name": "Mycotoxin Risk Kernel",
        "grade_impact": "Critical Biochemical Hazard",
        "cause": "Secondary fungal colonization resulting in toxic mycotoxic metabolite risk.",
        "action": "Strict quarantine; conduct ELISA chemical verification. Exclude from human food supply.",
    },
    "4_AP": {
        "name": "Aflatoxin Suspect Discoloration",
        "grade_impact": "High-Risk Biohazard",
        "cause": "Aspergillus flavus/parasiticus infestation under delayed field drying or warm moisture.",
        "action": "Reject lot for human or animal consumption; escalate quality audit.",
    },
    "5_BN": {
        "name": "Broken Grain (Kernels < 3/4 Length)",
        "grade_impact": "Milling Yield Deduction",
        "cause": "Excess moisture gradient stress-cracking or aggressive mechanical dehusking.",
        "action": "Calibrate rubber-roll huller clearance; check multi-pass polisher temperature.",
    },
    "6_UN": {
        "name": "Underdeveloped / Chalky Unclassified",
        "grade_impact": "Commercial Downgrade",
        "cause": "Incomplete starch grain endosperm filling; thermal stress during milky ripening stage.",
        "action": "Classify as Grade 3 / Feed grade; screen out with thickness sifter.",
    },
    "7_IM": {
        "name": "Immature / Greenish Grain",
        "grade_impact": "Maturity Defect",
        "cause": "Premature harvest of late-tillering panicles before physiological ripening.",
        "action": "Separate with length separator and aspirator; optimize future harvest date.",
    },
}


def translate_feature_name(feature_name: str) -> Dict[str, str]:
    """Translate a technical feature identifier into human-readable agronomic metadata."""
    if feature_name in FEATURE_AGRONOMIC_MAP:
        return FEATURE_AGRONOMIC_MAP[feature_name]

    if feature_name.startswith("deep_feature_"):
        dim = feature_name.split("_")[-1]
        return {
            "term": f"Deep Convolutional Pattern #{dim}",
            "category": "Deep Learned Representation",
            "description": f"High-level visual feature extracted by EfficientNet-B0 backbone (channel {dim}).",
        }

    # Fallback formatting
    clean_name = feature_name.replace("_", " ").title()
    return {
        "term": clean_name,
        "category": "Vision Descriptor",
        "description": f"Statistical computer vision feature: {feature_name}.",
    }


def generate_agronomic_diagnosis(
    predicted_class: str,
    confidence: float,
    positive_features: List[Tuple[str, float]],
    negative_features: List[Tuple[str, float]],
) -> Dict[str, Any]:
    """Produce a domain-specific narrative explaining the prediction in agricultural terms."""
    profile = CLASS_AGRONOMIC_PROFILES.get(
        predicted_class,
        {
            "name": predicted_class,
            "grade_impact": "General Inspection",
            "cause": "Unspecified grain anomaly.",
            "action": "Manual visual grading recommended.",
        },
    )

    translated_positives = []
    for feat, val in positive_features[:5]:
        meta = translate_feature_name(feat)
        translated_positives.append({
            "feature": feat,
            "term": meta["term"],
            "category": meta["category"],
            "impact": round(val, 4),
            "description": meta["description"],
        })

    key_descriptors = [p["term"] for p in translated_positives if p["category"] != "Deep Learned Representation"]
    if not key_descriptors:
        key_descriptors = [p["term"] for p in translated_positives]

    summary_sentence = (
        f"The system classified this grain as **{profile['name']}** ({predicted_class}) with "
        f"**{confidence:.1%}** confidence. Decision was strongly supported by "
        f"**{', '.join(key_descriptors[:3]) if key_descriptors else 'characteristic deep features'}**."
    )

    return {
        "class_name": profile["name"],
        "grade_impact": profile["grade_impact"],
        "probable_cause": profile["cause"],
        "recommended_action": profile["action"],
        "summary": summary_sentence,
        "key_agronomic_factors": translated_positives,
    }


__all__ = [
    "FEATURE_AGRONOMIC_MAP",
    "CLASS_AGRONOMIC_PROFILES",
    "translate_feature_name",
    "generate_agronomic_diagnosis",
]
