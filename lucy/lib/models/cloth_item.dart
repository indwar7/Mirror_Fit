enum ClothCategory {
  shirt,
  tShirt,
  jacket,
  coat,
  jeans,
  trousers,
  suit,
  kurta,
  shorts,
  sweatshirt,
  hoodie,
}

class ClothItem {
  final String id;
  final String name;
  final String description;
  final double price;
  final String imageUrl;
  final String? backImageUrl;
  final ClothCategory category;
  final List<String> sizes;
  final List<String> colors;
  final bool isNew;
  final bool isBestseller;

  const ClothItem({
    required this.id,
    required this.name,
    required this.description,
    required this.price,
    required this.imageUrl,
    this.backImageUrl,
    required this.category,
    this.sizes = const ['S', 'M', 'L', 'XL', 'XXL'],
    this.colors = const ['Navy', 'White', 'Black'],
    this.isNew = false,
    this.isBestseller = false,
  });

  String get fashnCategory {
    switch (category) {
      case ClothCategory.shirt:
      case ClothCategory.tShirt:
      case ClothCategory.jacket:
      case ClothCategory.coat:
      case ClothCategory.kurta:
      case ClothCategory.suit:
      case ClothCategory.sweatshirt:
      case ClothCategory.hoodie:
        return 'tops';
      case ClothCategory.jeans:
      case ClothCategory.trousers:
      case ClothCategory.shorts:
        return 'bottoms';
    }
  }

  String get categoryLabel {
    switch (category) {
      case ClothCategory.shirt:
        return 'Shirt';
      case ClothCategory.tShirt:
        return 'T-Shirt';
      case ClothCategory.jacket:
        return 'Jacket';
      case ClothCategory.coat:
        return 'Coat';
      case ClothCategory.jeans:
        return 'Jeans';
      case ClothCategory.trousers:
        return 'Trousers';
      case ClothCategory.suit:
        return 'Suit';
      case ClothCategory.kurta:
        return 'Kurta';
      case ClothCategory.shorts:
        return 'Shorts';
      case ClothCategory.sweatshirt:
        return 'Sweatshirt';
      case ClothCategory.hoodie:
        return 'Hoodie';
    }
  }
}

// ── Sample Catalog ───────────────────────────────────────────────────────────
// Replace imageUrl with real CDN or Firebase Storage URLs.
// The garment URL must be publicly accessible for Fashn.ai to process it.
class SampleCatalog {
  SampleCatalog._();

  // NOTE: Garment images MUST be garment-only shots (flat-lay, on hanger, or
  // mannequin). Photos of people wearing clothes will confuse the try-on AI.
  static const List<ClothItem> items = [
    ClothItem(
      id: 'ray_001',
      name: 'Classic White Shirt',
      description: 'Premium cotton formal shirt with a tailored fit. Perfect for boardroom meetings.',
      price: 2499,
      imageUrl: 'https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400',
      category: ClothCategory.shirt,
      sizes: ['S', 'M', 'L', 'XL', 'XXL'],
      colors: ['White', 'Ivory', 'Pearl'],
      isBestseller: true,
    ),
    ClothItem(
      id: 'ray_002',
      name: 'Navy Slim Fit Shirt',
      description: 'Contemporary slim-fit shirt in signature navy. Wrinkle-resistant fabric.',
      price: 2799,
      imageUrl: 'https://images.unsplash.com/photo-1591944489410-16ec1074a18e?w=400',
      category: ClothCategory.shirt,
      sizes: ['S', 'M', 'L', 'XL'],
      colors: ['Navy', 'Royal Blue', 'Indigo'],
      isNew: true,
    ),
    ClothItem(
      id: 'ray_003',
      name: 'Sky Blue Oxford',
      description: 'Button-down oxford shirt crafted from fine Egyptian cotton.',
      price: 3199,
      imageUrl: 'https://images.unsplash.com/photo-1732605559386-bc59426d1b16?w=400',
      category: ClothCategory.shirt,
      sizes: ['M', 'L', 'XL', 'XXL'],
      colors: ['Sky Blue', 'Light Blue', 'Chambray'],
    ),
    ClothItem(
      id: 'ray_004',
      name: 'Charcoal Blazer',
      description: 'Italian-style single-breasted blazer with premium wool blend.',
      price: 8999,
      imageUrl: 'https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=400',
      category: ClothCategory.jacket,
      sizes: ['M', 'L', 'XL'],
      colors: ['Charcoal', 'Dark Grey', 'Slate'],
      isBestseller: true,
    ),
    ClothItem(
      id: 'ray_005',
      name: 'Premium Wool Overcoat',
      description: 'Elegant full-length overcoat in premium Merino wool. Winter essential.',
      price: 14999,
      imageUrl: 'https://images.unsplash.com/photo-1608236159447-2d27116777f3?w=400',
      category: ClothCategory.coat,
      sizes: ['M', 'L', 'XL'],
      colors: ['Camel', 'Black', 'Navy'],
      isNew: true,
    ),
    ClothItem(
      id: 'ray_006',
      name: 'Black Crew Neck Tee',
      description: 'Premium supima cotton crew neck. Everyday luxury comfort.',
      price: 1499,
      imageUrl: 'https://images.unsplash.com/photo-1618354691373-d851c5c3a990?w=400',
      category: ClothCategory.tShirt,
      sizes: ['S', 'M', 'L', 'XL', 'XXL'],
      colors: ['Black', 'Charcoal', 'Jet'],
    ),
    ClothItem(
      id: 'ray_007',
      name: 'Olive Polo',
      description: 'Classic polo with embroidered logo. Pique cotton fabric.',
      price: 1999,
      imageUrl: 'https://images.unsplash.com/photo-1586363129094-d7a38564fae1?w=400',
      category: ClothCategory.tShirt,
      sizes: ['S', 'M', 'L', 'XL'],
      colors: ['Olive', 'Forest Green', 'Sage'],
      isNew: true,
    ),
    ClothItem(
      id: 'ray_008',
      name: 'Indigo Slim Jeans',
      description: 'Japanese selvedge denim with a modern slim taper.',
      price: 3999,
      imageUrl: 'https://images.unsplash.com/photo-1659167099846-a0dbfc52aa2d?w=400',
      category: ClothCategory.jeans,
      sizes: ['30', '32', '34', '36', '38'],
      colors: ['Indigo', 'Dark Wash', 'Raw'],
      isBestseller: true,
    ),
    ClothItem(
      id: 'ray_009',
      name: 'Beige Chinos',
      description: 'Tailored chino trousers with stretch comfort. Business casual essential.',
      price: 2999,
      imageUrl: 'https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=400',
      category: ClothCategory.trousers,
      sizes: ['30', '32', '34', '36'],
      colors: ['Beige', 'Khaki', 'Sand'],
    ),
    ClothItem(
      id: 'ray_010',
      name: 'Navy Two-Piece Suit',
      description: 'Signature suit in Italian wool. Wedding & celebration ready.',
      price: 18999,
      imageUrl: 'https://images.unsplash.com/photo-1598808503746-f34c53b9323e?w=400',
      category: ClothCategory.suit,
      sizes: ['38', '40', '42', '44'],
      colors: ['Navy', 'Midnight Blue', 'Classic Navy'],
      isBestseller: true,
    ),
    ClothItem(
      id: 'ray_011',
      name: 'Silk Embroidered Kurta',
      description: 'Festive silk kurta with intricate thread embroidery. For the modern Indian man.',
      price: 6999,
      imageUrl: 'https://images.unsplash.com/photo-1760287364328-e30221615f2e?w=400',
      category: ClothCategory.kurta,
      sizes: ['S', 'M', 'L', 'XL'],
      colors: ['Ivory', 'Gold', 'Maroon'],
      isNew: true,
    ),
    ClothItem(
      id: 'ray_012',
      name: 'Tan Leather Jacket',
      description: 'Hand-finished lambskin leather jacket. Timeless weekend style.',
      price: 12999,
      imageUrl: 'https://images.unsplash.com/photo-1623854156816-4c4fc355ffc7?w=400',
      category: ClothCategory.jacket,
      sizes: ['M', 'L', 'XL'],
      colors: ['Tan', 'Cognac', 'Brown'],
    ),
    ClothItem(
      id: 'ray_013',
      name: 'Olive Cargo Shorts',
      description: 'Relaxed-fit cargo shorts with utility pockets. Built for summer adventures.',
      price: 1799,
      imageUrl: 'https://images.unsplash.com/photo-1565084888279-aca607ecce0c?w=400',
      category: ClothCategory.shorts,
      sizes: ['28', '30', '32', '34', '36'],
      colors: ['Olive', 'Khaki', 'Black'],
      isNew: true,
    ),
    ClothItem(
      id: 'ray_014',
      name: 'Navy Chino Shorts',
      description: 'Smart chino shorts in a clean navy finish. Office-to-weekend versatility.',
      price: 1599,
      imageUrl: 'https://images.unsplash.com/photo-1591195853828-11db59a44f43?w=400',
      category: ClothCategory.shorts,
      sizes: ['28', '30', '32', '34'],
      colors: ['Navy', 'Beige', 'Charcoal'],
    ),
    ClothItem(
      id: 'ray_015',
      name: 'Graphite Hoodie',
      description: 'French terry cotton hoodie with a brushed interior. Soft, warm, essential.',
      price: 2999,
      imageUrl: 'https://images.unsplash.com/photo-1556821840-3a63f15732ce?w=400',
      category: ClothCategory.hoodie,
      sizes: ['S', 'M', 'L', 'XL', 'XXL'],
      colors: ['Charcoal', 'Black', 'Slate'],
      isBestseller: true,
    ),
    ClothItem(
      id: 'ray_016',
      name: 'Cream Pullover Sweatshirt',
      description: 'Relaxed-fit fleece sweatshirt with dropped shoulders. Effortless off-duty look.',
      price: 2499,
      imageUrl: 'https://images.unsplash.com/photo-1620799140408-edc6dcb6d633?w=400',
      category: ClothCategory.sweatshirt,
      sizes: ['S', 'M', 'L', 'XL'],
      colors: ['Ivory', 'Sand', 'Sage'],
      isNew: true,
    ),
    ClothItem(
      id: 'ray_017',
      name: 'White Denim Shorts',
      description: 'Clean-cut denim shorts with a slim silhouette. Fresh summer essential.',
      price: 2199,
      imageUrl: 'https://images.unsplash.com/photo-1542272604-787c3835535d?w=400',
      category: ClothCategory.shorts,
      sizes: ['28', '30', '32', '34'],
      colors: ['White', 'Light Blue', 'Ivory'],
    ),
    ClothItem(
      id: 'ray_018',
      name: 'Maroon Graphic Tee',
      description: 'Heavyweight jersey tee with a vintage washed finish. Expressive street style.',
      price: 1299,
      imageUrl: 'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400',
      category: ClothCategory.tShirt,
      sizes: ['S', 'M', 'L', 'XL', 'XXL'],
      colors: ['Maroon', 'Black', 'Navy'],
      isNew: true,
    ),
    ClothItem(
      id: 'ray_019',
      name: 'Camel Trench Coat',
      description: 'Double-breasted trench in water-resistant gabardine. The definitive outerwear piece.',
      price: 16999,
      imageUrl: 'https://images.unsplash.com/photo-1539533113208-f6df8cc8b543?w=400',
      category: ClothCategory.coat,
      sizes: ['M', 'L', 'XL'],
      colors: ['Camel', 'Beige', 'Black'],
      isBestseller: true,
    ),
    ClothItem(
      id: 'ray_020',
      name: 'Striped Linen Shirt',
      description: 'Breathable linen-blend shirt with subtle stripe. Holiday and resort ready.',
      price: 2799,
      imageUrl: 'https://images.unsplash.com/photo-1602810318383-e386cc2a3ccf?w=400',
      category: ClothCategory.shirt,
      sizes: ['S', 'M', 'L', 'XL'],
      colors: ['White', 'Sky Blue', 'Beige'],
      isNew: true,
    ),
  ];

  static List<ClothItem> byCategory(ClothCategory category) =>
      items.where((item) => item.category == category).toList();

  static List<ClothItem> get bestsellers =>
      items.where((item) => item.isBestseller).toList();

  static List<ClothItem> get newArrivals =>
      items.where((item) => item.isNew).toList();
}
