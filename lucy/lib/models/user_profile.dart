class UserProfile {
  final String name;
  final double heightCm;
  final double weightKg;
  final BodyType bodyType;
  final SkinTone skinTone;
  final List<StylePreference> stylePrefs;
  final Occasion primaryOccasion;

  const UserProfile({
    this.name = '',
    this.heightCm = 170,
    this.weightKg = 70,
    this.bodyType = BodyType.average,
    this.skinTone = SkinTone.medium,
    this.stylePrefs = const [StylePreference.casual],
    this.primaryOccasion = Occasion.everyday,
  });

  UserProfile copyWith({
    String? name,
    double? heightCm,
    double? weightKg,
    BodyType? bodyType,
    SkinTone? skinTone,
    List<StylePreference>? stylePrefs,
    Occasion? primaryOccasion,
  }) {
    return UserProfile(
      name: name ?? this.name,
      heightCm: heightCm ?? this.heightCm,
      weightKg: weightKg ?? this.weightKg,
      bodyType: bodyType ?? this.bodyType,
      skinTone: skinTone ?? this.skinTone,
      stylePrefs: stylePrefs ?? this.stylePrefs,
      primaryOccasion: primaryOccasion ?? this.primaryOccasion,
    );
  }

  String get recommendedSize {
    final bmi = weightKg / ((heightCm / 100) * (heightCm / 100));
    if (bmi < 18.5) return 'S';
    if (bmi < 22) return 'M';
    if (bmi < 25) return 'L';
    if (bmi < 28) return 'XL';
    return 'XXL';
  }

  String get recommendedPantSize {
    if (weightKg < 55) return '28';
    if (weightKg < 65) return '30';
    if (weightKg < 75) return '32';
    if (weightKg < 85) return '34';
    if (weightKg < 95) return '36';
    return '38';
  }

  Map<String, dynamic> toJson() => {
        'name': name,
        'heightCm': heightCm,
        'weightKg': weightKg,
        'bodyType': bodyType.index,
        'skinTone': skinTone.index,
        'stylePrefs': stylePrefs.map((e) => e.index).toList(),
        'primaryOccasion': primaryOccasion.index,
      };

  factory UserProfile.fromJson(Map<String, dynamic> json) => UserProfile(
        name: json['name'] as String? ?? '',
        heightCm: (json['heightCm'] as num?)?.toDouble() ?? 170,
        weightKg: (json['weightKg'] as num?)?.toDouble() ?? 70,
        bodyType: BodyType.values[json['bodyType'] as int? ?? 1],
        skinTone: SkinTone.values[json['skinTone'] as int? ?? 2],
        stylePrefs: (json['stylePrefs'] as List?)
                ?.map((e) => StylePreference.values[e as int])
                .toList() ??
            [StylePreference.casual],
        primaryOccasion:
            Occasion.values[json['primaryOccasion'] as int? ?? 0],
      );
}

enum BodyType {
  slim('Slim', 'Lean build'),
  average('Average', 'Medium build'),
  athletic('Athletic', 'Muscular build'),
  broad('Broad', 'Wider frame');

  final String label;
  final String description;
  const BodyType(this.label, this.description);
}

enum SkinTone {
  fair('Fair', 0xFFFDE7C7),
  light('Light', 0xFFE8C9A0),
  medium('Medium', 0xFFD4A574),
  olive('Olive', 0xFFC19A6B),
  brown('Brown', 0xFF8D5524),
  dark('Dark', 0xFF5C3317);

  final String label;
  final int colorValue;
  const SkinTone(this.label, this.colorValue);
}

enum StylePreference {
  formal('Formal', 'Suits, shirts, ties'),
  casual('Casual', 'T-shirts, jeans, chinos'),
  ethnic('Ethnic', 'Kurtas, sherwanis'),
  streetwear('Streetwear', 'Bold, trendy looks'),
  minimal('Minimal', 'Clean, understated');

  final String label;
  final String description;
  const StylePreference(this.label, this.description);
}

enum Occasion {
  everyday('Everyday'),
  office('Office'),
  wedding('Wedding'),
  party('Party'),
  festival('Festival');

  final String label;
  const Occasion(this.label);
}
