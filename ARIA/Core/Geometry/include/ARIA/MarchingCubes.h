#pragma once

#include "ARIA/Math.h"
#include "ARIA/Vec.h"

namespace ARIA {

ARIA_CONST static inline constexpr int8 vtkMarchingCubesTriangleCases[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},  // 0 0
    {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     // 1 1
    {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     // 2 1
    {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},        // 3 2
    {1, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    // 4 1
    {0, 3, 8, 1, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 5 3
    {9, 11, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 6 2
    {2, 3, 8, 2, 8, 11, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},         // 7 5
    {3, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    // 8 1
    {0, 2, 10, 8, 0, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},      // 9 2
    {1, 0, 9, 2, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 10 3
    {1, 2, 10, 1, 10, 9, 9, 10, 8, -1, -1, -1, -1, -1, -1, -1},        // 11 5
    {3, 1, 11, 10, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     // 12 2
    {0, 1, 11, 0, 11, 8, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},       // 13 5
    {3, 0, 9, 3, 9, 10, 10, 9, 11, -1, -1, -1, -1, -1, -1, -1},        // 14 5
    {9, 11, 8, 11, 10, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     // 15 8
    {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     // 16 1
    {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},        // 17 2
    {0, 9, 1, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},        // 18 3
    {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},           // 19 5
    {1, 11, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 20 4
    {3, 7, 4, 3, 4, 0, 1, 11, 2, -1, -1, -1, -1, -1, -1, -1},          // 21 7
    {9, 11, 2, 9, 2, 0, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1},          // 22 7
    {2, 9, 11, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},             // 23 14
    {8, 7, 4, 3, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 24 3
    {10, 7, 4, 10, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},         // 25 5
    {9, 1, 0, 8, 7, 4, 2, 10, 3, -1, -1, -1, -1, -1, -1, -1},          // 26 6
    {4, 10, 7, 9, 10, 4, 9, 2, 10, 9, 1, 2, -1, -1, -1, -1},           // 27 9
    {3, 1, 11, 3, 11, 10, 7, 4, 8, -1, -1, -1, -1, -1, -1, -1},        // 28 7
    {1, 11, 10, 1, 10, 4, 1, 4, 0, 7, 4, 10, -1, -1, -1, -1},          // 29 11
    {4, 8, 7, 9, 10, 0, 9, 11, 10, 10, 3, 0, -1, -1, -1, -1},          // 30 12
    {4, 10, 7, 4, 9, 10, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},       // 31 5
    {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     // 32 1
    {9, 4, 5, 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},        // 33 3
    {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},        // 34 2
    {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},           // 35 5
    {1, 11, 2, 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 36 3
    {3, 8, 0, 1, 11, 2, 4, 5, 9, -1, -1, -1, -1, -1, -1, -1},          // 37 6
    {5, 11, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},          // 38 5
    {2, 5, 11, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},             // 39 9
    {9, 4, 5, 2, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 40 4
    {0, 2, 10, 0, 10, 8, 4, 5, 9, -1, -1, -1, -1, -1, -1, -1},         // 41 7
    {0, 4, 5, 0, 5, 1, 2, 10, 3, -1, -1, -1, -1, -1, -1, -1},          // 42 7
    {2, 5, 1, 2, 8, 5, 2, 10, 8, 4, 5, 8, -1, -1, -1, -1},             // 43 11
    {11, 10, 3, 11, 3, 1, 9, 4, 5, -1, -1, -1, -1, -1, -1, -1},        // 44 7
    {4, 5, 9, 0, 1, 8, 8, 1, 11, 8, 11, 10, -1, -1, -1, -1},           // 45 12
    {5, 0, 4, 5, 10, 0, 5, 11, 10, 10, 3, 0, -1, -1, -1, -1},          // 46 14
    {5, 8, 4, 5, 11, 8, 11, 10, 8, -1, -1, -1, -1, -1, -1, -1},        // 47 5
    {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},        // 48 2
    {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},           // 49 5
    {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},           // 50 5
    {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},        // 51 8
    {9, 8, 7, 9, 7, 5, 11, 2, 1, -1, -1, -1, -1, -1, -1, -1},          // 52 7
    {11, 2, 1, 9, 0, 5, 5, 0, 3, 5, 3, 7, -1, -1, -1, -1},             // 53 12
    {8, 2, 0, 8, 5, 2, 8, 7, 5, 11, 2, 5, -1, -1, -1, -1},             // 54 11
    {2, 5, 11, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},          // 55 5
    {7, 5, 9, 7, 9, 8, 3, 2, 10, -1, -1, -1, -1, -1, -1, -1},          // 56 7
    {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 10, 7, -1, -1, -1, -1},             // 57 14
    {2, 10, 3, 0, 8, 1, 1, 8, 7, 1, 7, 5, -1, -1, -1, -1},             // 58 12
    {10, 1, 2, 10, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},         // 59 5
    {9, 8, 5, 8, 7, 5, 11, 3, 1, 11, 10, 3, -1, -1, -1, -1},           // 60 10
    {5, 0, 7, 5, 9, 0, 7, 0, 10, 1, 11, 0, 10, 0, 11, -1},             // 61 7
    {10, 0, 11, 10, 3, 0, 11, 0, 5, 8, 7, 0, 5, 0, 7, -1},             // 62 7
    {10, 5, 11, 7, 5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     // 63 2
    {11, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    // 64 1
    {0, 3, 8, 5, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 65 4
    {9, 1, 0, 5, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 66 3
    {1, 3, 8, 1, 8, 9, 5, 6, 11, -1, -1, -1, -1, -1, -1, -1},          // 67 7
    {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},        // 68 2
    {1, 5, 6, 1, 6, 2, 3, 8, 0, -1, -1, -1, -1, -1, -1, -1},           // 69 7
    {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},           // 70 5
    {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},              // 71 11
    {2, 10, 3, 11, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},      // 72 3
    {10, 8, 0, 10, 0, 2, 11, 5, 6, -1, -1, -1, -1, -1, -1, -1},        // 73 7
    {0, 9, 1, 2, 10, 3, 5, 6, 11, -1, -1, -1, -1, -1, -1, -1},         // 74 6
    {5, 6, 11, 1, 2, 9, 9, 2, 10, 9, 10, 8, -1, -1, -1, -1},           // 75 12
    {6, 10, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},          // 76 5
    {0, 10, 8, 0, 5, 10, 0, 1, 5, 5, 6, 10, -1, -1, -1, -1},           // 77 14
    {3, 6, 10, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},             // 78 9
    {6, 9, 5, 6, 10, 9, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},         // 79 5
    {5, 6, 11, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 80 3
    {4, 0, 3, 4, 3, 7, 6, 11, 5, -1, -1, -1, -1, -1, -1, -1},          // 81 7
    {1, 0, 9, 5, 6, 11, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1},          // 82 6
    {11, 5, 6, 1, 7, 9, 1, 3, 7, 7, 4, 9, -1, -1, -1, -1},             // 83 12
    {6, 2, 1, 6, 1, 5, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1},           // 84 7
    {1, 5, 2, 5, 6, 2, 3, 4, 0, 3, 7, 4, -1, -1, -1, -1},              // 85 10
    {8, 7, 4, 9, 5, 0, 0, 5, 6, 0, 6, 2, -1, -1, -1, -1},              // 86 12
    {7, 9, 3, 7, 4, 9, 3, 9, 2, 5, 6, 9, 2, 9, 6, -1},                 // 87 7
    {3, 2, 10, 7, 4, 8, 11, 5, 6, -1, -1, -1, -1, -1, -1, -1},         // 88 6
    {5, 6, 11, 4, 2, 7, 4, 0, 2, 2, 10, 7, -1, -1, -1, -1},            // 89 12
    {0, 9, 1, 4, 8, 7, 2, 10, 3, 5, 6, 11, -1, -1, -1, -1},            // 90 13
    {9, 1, 2, 9, 2, 10, 9, 10, 4, 7, 4, 10, 5, 6, 11, -1},             // 91 6
    {8, 7, 4, 3, 5, 10, 3, 1, 5, 5, 6, 10, -1, -1, -1, -1},            // 92 12
    {5, 10, 1, 5, 6, 10, 1, 10, 0, 7, 4, 10, 0, 10, 4, -1},            // 93 7
    {0, 9, 5, 0, 5, 6, 0, 6, 3, 10, 3, 6, 8, 7, 4, -1},                // 94 6
    {6, 9, 5, 6, 10, 9, 4, 9, 7, 7, 9, 10, -1, -1, -1, -1},            // 95 3
    {11, 9, 4, 6, 11, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},      // 96 2
    {4, 6, 11, 4, 11, 9, 0, 3, 8, -1, -1, -1, -1, -1, -1, -1},         // 97 7
    {11, 1, 0, 11, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},         // 98 5
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 11, 1, -1, -1, -1, -1},             // 99 14
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},           // 100 5
    {3, 8, 0, 1, 9, 2, 2, 9, 4, 2, 4, 6, -1, -1, -1, -1},              // 101 12
    {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},        // 102 8
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},           // 103 5
    {11, 9, 4, 11, 4, 6, 10, 3, 2, -1, -1, -1, -1, -1, -1, -1},        // 104 7
    {0, 2, 8, 2, 10, 8, 4, 11, 9, 4, 6, 11, -1, -1, -1, -1},           // 105 10
    {3, 2, 10, 0, 6, 1, 0, 4, 6, 6, 11, 1, -1, -1, -1, -1},            // 106 12
    {6, 1, 4, 6, 11, 1, 4, 1, 8, 2, 10, 1, 8, 1, 10, -1},              // 107 7
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 10, 3, 6, -1, -1, -1, -1},             // 108 11
    {8, 1, 10, 8, 0, 1, 10, 1, 6, 9, 4, 1, 6, 1, 4, -1},               // 109 7
    {3, 6, 10, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},          // 110 5
    {6, 8, 4, 10, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 111 2
    {7, 6, 11, 7, 11, 8, 8, 11, 9, -1, -1, -1, -1, -1, -1, -1},        // 112 5
    {0, 3, 7, 0, 7, 11, 0, 11, 9, 6, 11, 7, -1, -1, -1, -1},           // 113 11
    {11, 7, 6, 1, 7, 11, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},            // 114 9
    {11, 7, 6, 11, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},         // 115 5
    {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},              // 116 14
    {2, 9, 6, 2, 1, 9, 6, 9, 7, 0, 3, 9, 7, 9, 3, -1},                 // 117 7
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},           // 118 5
    {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},        // 119 2
    {2, 10, 3, 11, 8, 6, 11, 9, 8, 8, 7, 6, -1, -1, -1, -1},           // 120 12
    {2, 7, 0, 2, 10, 7, 0, 7, 9, 6, 11, 7, 9, 7, 11, -1},              // 121 7
    {1, 0, 8, 1, 8, 7, 1, 7, 11, 6, 11, 7, 2, 10, 3, -1},              // 122 6
    {10, 1, 2, 10, 7, 1, 11, 1, 6, 6, 1, 7, -1, -1, -1, -1},           // 123 3
    {8, 6, 9, 8, 7, 6, 9, 6, 1, 10, 3, 6, 1, 6, 3, -1},                // 124 7
    {0, 1, 9, 10, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 125 4
    {7, 0, 8, 7, 6, 0, 3, 0, 10, 10, 0, 6, -1, -1, -1, -1},            // 126 3
    {7, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    // 127 1
    {7, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    // 128 1
    {3, 8, 0, 10, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 129 3
    {0, 9, 1, 10, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 130 4
    {8, 9, 1, 8, 1, 3, 10, 6, 7, -1, -1, -1, -1, -1, -1, -1},          // 131 7
    {11, 2, 1, 6, 7, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},      // 132 3
    {1, 11, 2, 3, 8, 0, 6, 7, 10, -1, -1, -1, -1, -1, -1, -1},         // 133 6
    {2, 0, 9, 2, 9, 11, 6, 7, 10, -1, -1, -1, -1, -1, -1, -1},         // 134 7
    {6, 7, 10, 2, 3, 11, 11, 3, 8, 11, 8, 9, -1, -1, -1, -1},          // 135 12
    {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},        // 136 2
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},           // 137 5
    {2, 6, 7, 2, 7, 3, 0, 9, 1, -1, -1, -1, -1, -1, -1, -1},           // 138 7
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},              // 139 14
    {11, 6, 7, 11, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},         // 140 5
    {11, 6, 7, 1, 11, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},            // 141 9
    {0, 7, 3, 0, 11, 7, 0, 9, 11, 6, 7, 11, -1, -1, -1, -1},           // 142 11
    {7, 11, 6, 7, 8, 11, 8, 9, 11, -1, -1, -1, -1, -1, -1, -1},        // 143 5
    {6, 4, 8, 10, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 144 2
    {3, 10, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},          // 145 5
    {8, 10, 6, 8, 6, 4, 9, 1, 0, -1, -1, -1, -1, -1, -1, -1},          // 146 7
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 10, 6, 3, -1, -1, -1, -1},             // 147 11
    {6, 4, 8, 6, 8, 10, 2, 1, 11, -1, -1, -1, -1, -1, -1, -1},         // 148 7
    {1, 11, 2, 3, 10, 0, 0, 10, 6, 0, 6, 4, -1, -1, -1, -1},           // 149 12
    {4, 8, 10, 4, 10, 6, 0, 9, 2, 2, 9, 11, -1, -1, -1, -1},           // 150 10
    {11, 3, 9, 11, 2, 3, 9, 3, 4, 10, 6, 3, 4, 3, 6, -1},              // 151 7
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},           // 152 5
    {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},        // 153 8
    {1, 0, 9, 2, 4, 3, 2, 6, 4, 4, 8, 3, -1, -1, -1, -1},              // 154 12
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},           // 155 5
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 11, -1, -1, -1, -1},             // 156 14
    {11, 0, 1, 11, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},         // 157 5
    {4, 3, 6, 4, 8, 3, 6, 3, 11, 0, 9, 3, 11, 3, 9, -1},               // 158 7
    {11, 4, 9, 6, 4, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},      // 159 2
    {4, 5, 9, 7, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 160 3
    {0, 3, 8, 4, 5, 9, 10, 6, 7, -1, -1, -1, -1, -1, -1, -1},          // 161 6
    {5, 1, 0, 5, 0, 4, 7, 10, 6, -1, -1, -1, -1, -1, -1, -1},          // 162 7
    {10, 6, 7, 8, 4, 3, 3, 4, 5, 3, 5, 1, -1, -1, -1, -1},             // 163 12
    {9, 4, 5, 11, 2, 1, 7, 10, 6, -1, -1, -1, -1, -1, -1, -1},         // 164 6
    {6, 7, 10, 1, 11, 2, 0, 3, 8, 4, 5, 9, -1, -1, -1, -1},            // 165 13
    {7, 10, 6, 5, 11, 4, 4, 11, 2, 4, 2, 0, -1, -1, -1, -1},           // 166 12
    {3, 8, 4, 3, 4, 5, 3, 5, 2, 11, 2, 5, 10, 6, 7, -1},               // 167 6
    {7, 3, 2, 7, 2, 6, 5, 9, 4, -1, -1, -1, -1, -1, -1, -1},           // 168 7
    {9, 4, 5, 0, 6, 8, 0, 2, 6, 6, 7, 8, -1, -1, -1, -1},              // 169 12
    {3, 2, 6, 3, 6, 7, 1, 0, 5, 5, 0, 4, -1, -1, -1, -1},              // 170 10
    {6, 8, 2, 6, 7, 8, 2, 8, 1, 4, 5, 8, 1, 8, 5, -1},                 // 171 7
    {9, 4, 5, 11, 6, 1, 1, 6, 7, 1, 7, 3, -1, -1, -1, -1},             // 172 12
    {1, 11, 6, 1, 6, 7, 1, 7, 0, 8, 0, 7, 9, 4, 5, -1},                // 173 6
    {4, 11, 0, 4, 5, 11, 0, 11, 3, 6, 7, 11, 3, 11, 7, -1},            // 174 7
    {7, 11, 6, 7, 8, 11, 5, 11, 4, 4, 11, 8, -1, -1, -1, -1},          // 175 3
    {6, 5, 9, 6, 9, 10, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},         // 176 5
    {3, 10, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},             // 177 9
    {0, 8, 10, 0, 10, 5, 0, 5, 1, 5, 10, 6, -1, -1, -1, -1},           // 178 14
    {6, 3, 10, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},          // 179 5
    {1, 11, 2, 9, 10, 5, 9, 8, 10, 10, 6, 5, -1, -1, -1, -1},          // 180 12
    {0, 3, 10, 0, 10, 6, 0, 6, 9, 5, 9, 6, 1, 11, 2, -1},              // 181 6
    {10, 5, 8, 10, 6, 5, 8, 5, 0, 11, 2, 5, 0, 5, 2, -1},              // 182 7
    {6, 3, 10, 6, 5, 3, 2, 3, 11, 11, 3, 5, -1, -1, -1, -1},           // 183 3
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},              // 184 11
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},           // 185 5
    {1, 8, 5, 1, 0, 8, 5, 8, 6, 3, 2, 8, 6, 8, 2, -1},                 // 186 7
    {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},        // 187 2
    {1, 6, 3, 1, 11, 6, 3, 6, 8, 5, 9, 6, 8, 6, 9, -1},                // 188 7
    {11, 0, 1, 11, 6, 0, 9, 0, 5, 5, 0, 6, -1, -1, -1, -1},            // 189 3
    {0, 8, 3, 5, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 190 4
    {11, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    // 191 1
    {10, 11, 5, 7, 10, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     // 192 2
    {10, 11, 5, 10, 5, 7, 8, 0, 3, -1, -1, -1, -1, -1, -1, -1},        // 193 7
    {5, 7, 10, 5, 10, 11, 1, 0, 9, -1, -1, -1, -1, -1, -1, -1},        // 194 7
    {11, 5, 7, 11, 7, 10, 9, 1, 8, 8, 1, 3, -1, -1, -1, -1},           // 195 10
    {10, 2, 1, 10, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},         // 196 5
    {0, 3, 8, 1, 7, 2, 1, 5, 7, 7, 10, 2, -1, -1, -1, -1},             // 197 12
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 10, -1, -1, -1, -1},             // 198 14
    {7, 2, 5, 7, 10, 2, 5, 2, 9, 3, 8, 2, 9, 2, 8, -1},                // 199 7
    {2, 11, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},          // 200 5
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 11, 5, 2, -1, -1, -1, -1},             // 201 11
    {9, 1, 0, 5, 3, 11, 5, 7, 3, 3, 2, 11, -1, -1, -1, -1},            // 202 12
    {9, 2, 8, 9, 1, 2, 8, 2, 7, 11, 5, 2, 7, 2, 5, -1},                // 203 7
    {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},        // 204 8
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},           // 205 5
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},           // 206 5
    {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},        // 207 2
    {5, 4, 8, 5, 8, 11, 11, 8, 10, -1, -1, -1, -1, -1, -1, -1},        // 208 5
    {5, 4, 0, 5, 0, 10, 5, 10, 11, 10, 0, 3, -1, -1, -1, -1},          // 209 14
    {0, 9, 1, 8, 11, 4, 8, 10, 11, 11, 5, 4, -1, -1, -1, -1},          // 210 12
    {11, 4, 10, 11, 5, 4, 10, 4, 3, 9, 1, 4, 3, 4, 1, -1},             // 211 7
    {2, 1, 5, 2, 5, 8, 2, 8, 10, 4, 8, 5, -1, -1, -1, -1},             // 212 11
    {0, 10, 4, 0, 3, 10, 4, 10, 5, 2, 1, 10, 5, 10, 1, -1},            // 213 7
    {0, 5, 2, 0, 9, 5, 2, 5, 10, 4, 8, 5, 10, 5, 8, -1},               // 214 7
    {9, 5, 4, 2, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 215 4
    {2, 11, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},             // 216 9
    {5, 2, 11, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},          // 217 5
    {3, 2, 11, 3, 11, 5, 3, 5, 8, 4, 8, 5, 0, 9, 1, -1},               // 218 6
    {5, 2, 11, 5, 4, 2, 1, 2, 9, 9, 2, 4, -1, -1, -1, -1},             // 219 3
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},           // 220 5
    {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},        // 221 2
    {8, 5, 4, 8, 3, 5, 9, 5, 0, 0, 5, 3, -1, -1, -1, -1},              // 222 3
    {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     // 223 1
    {4, 7, 10, 4, 10, 9, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},       // 224 5
    {0, 3, 8, 4, 7, 9, 9, 7, 10, 9, 10, 11, -1, -1, -1, -1},           // 225 12
    {1, 10, 11, 1, 4, 10, 1, 0, 4, 7, 10, 4, -1, -1, -1, -1},          // 226 11
    {3, 4, 1, 3, 8, 4, 1, 4, 11, 7, 10, 4, 11, 4, 10, -1},             // 227 7
    {4, 7, 10, 9, 4, 10, 9, 10, 2, 9, 2, 1, -1, -1, -1, -1},           // 228 9
    {9, 4, 7, 9, 7, 10, 9, 10, 1, 2, 1, 10, 0, 3, 8, -1},              // 229 6
    {10, 4, 7, 10, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},         // 230 5
    {10, 4, 7, 10, 2, 4, 8, 4, 3, 3, 4, 2, -1, -1, -1, -1},            // 231 3
    {2, 11, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},             // 232 14
    {9, 7, 11, 9, 4, 7, 11, 7, 2, 8, 0, 7, 2, 7, 0, -1},               // 233 7
    {3, 11, 7, 3, 2, 11, 7, 11, 4, 1, 0, 11, 4, 11, 0, -1},            // 234 7
    {1, 2, 11, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 235 4
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},           // 236 5
    {4, 1, 9, 4, 7, 1, 0, 1, 8, 8, 1, 7, -1, -1, -1, -1},              // 237 3
    {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},        // 238 2
    {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     // 239 1
    {9, 8, 11, 11, 8, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     // 240 8
    {3, 9, 0, 3, 10, 9, 10, 11, 9, -1, -1, -1, -1, -1, -1, -1},        // 241 5
    {0, 11, 1, 0, 8, 11, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},       // 242 5
    {3, 11, 1, 10, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     // 243 2
    {1, 10, 2, 1, 9, 10, 9, 8, 10, -1, -1, -1, -1, -1, -1, -1},        // 244 5
    {3, 9, 0, 3, 10, 9, 1, 9, 2, 2, 9, 10, -1, -1, -1, -1},            // 245 3
    {0, 10, 2, 8, 10, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},      // 246 2
    {3, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    // 247 1
    {2, 8, 3, 2, 11, 8, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},         // 248 5
    {9, 2, 11, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},       // 249 2
    {2, 8, 3, 2, 11, 8, 0, 8, 1, 1, 8, 11, -1, -1, -1, -1},            // 250 3
    {1, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},    // 251 1
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},        // 252 2
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     // 253 1
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},     // 254 1
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}}; // 255 0

ARIA_CONST static inline constexpr int8 vtkMarchingCubes_edges[12][2] = {
    {0, 1}, {1, 2}, {3, 2}, {0, 3}, {4, 5}, {5, 6}, {7, 6}, {4, 7}, {0, 4}, {1, 5}, {3, 7}, {2, 6}};

ARIA_CONST static inline constexpr int8 vtkMarchingSquaresLineCases[16][5] = {
    {-1, -1, -1, -1, -1}, {0, 3, -1, -1, -1}, {1, 0, -1, -1, -1}, {1, 3, -1, -1, -1},
    {2, 1, -1, -1, -1},   {0, 3, 2, 1, -1},   {2, 0, -1, -1, -1}, {2, 3, -1, -1, -1},
    {3, 2, -1, -1, -1},   {0, 2, -1, -1, -1}, {1, 0, 3, 2, -1},   {1, 2, -1, -1, -1},
    {3, 1, -1, -1, -1},   {0, 1, -1, -1, -1}, {3, 0, -1, -1, -1}, {-1, -1, -1, -1, -1}};

ARIA_CONST static inline constexpr int8 vtkMarchingSquares_edges[4][2] = {{0, 1}, {1, 3}, {2, 3}, {0, 2}};

//
//
//
template <uint dim>
[[nodiscard]] ARIA_HOST_DEVICE static consteval auto MarchingCubesCases() {
  if constexpr (dim == 2)
    return vtkMarchingSquaresLineCases;
  else if constexpr (dim == 3)
    return vtkMarchingCubesTriangleCases;
}

template <uint dim>
[[nodiscard]] ARIA_HOST_DEVICE static consteval auto MarchingCubes_edges() {
  if constexpr (dim == 2)
    return vtkMarchingSquares_edges;
  else if constexpr (dim == 3)
    return vtkMarchingCubes_edges;
}

//
//
//
//
//
// Compute $x^N$ at compile time.
template <int N, typename T>
ARIA_HOST_DEVICE static T consteval powN(T x) {
  static_assert(N >= 0);

  if constexpr (N > 0) {
    return x * powN<N - 1>(x);
  } else {
    return 1;
  }
}

//
//
//
//
//
template <uint dim>
class DualVertex {
public:
  DualVertex() = default;
  ARIA_COPY_MOVE_ABILITY(DualVertex, default, default);

public:
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, pos, pos_);
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, value, value_);

private:
  Vec<Real, dim> pos_;
  Real value_;
};

//
//
//
template <uint dim>
class Cell {
public:
  Cell() = default;
  ARIA_COPY_MOVE_ABILITY(Cell, default, default);

public:
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, coord, coord_);
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, value, value_);

public:
  [[nodiscard]] ARIA_HOST_DEVICE Vec<Real, dim> center() const { return coord_.template cast<Real>(); }

  [[nodiscard]] ARIA_HOST_DEVICE DualVertex<dim> dualVertex() const {
    DualVertex<dim> v;
    v.pos() = center();
    v.value() = value();
    return v;
  }

private:
  Vec<int, dim> coord_;
  Real value_;
};

//
//
//
//
//
template <uint dim>
class MarchingCubes {
public:
private:
  template <typename F>
  ARIA_HOST_DEVICE static void
  DoMarchingCubesOn(const Vec<int, dim> &mirror, const auto &zOrder, Real isoValue, F &&f) {
    // we have OUR cells in z-order, but VTK case table assumes
    // everything is is VTK 'hexahedron' ordering, so let's rearrange
    // ... and while doing so, also make sure that we flip based on
    // which direction the parent cell created this dual from
    DualVertex<dim> vertex[powN<dim>(2)];
    if constexpr (dim == 2) {
      vertex = {zOrder[0 + mirror.y()][0 + mirror.x()].dualVertex(), //
                zOrder[0 + mirror.y()][1 - mirror.x()].dualVertex(), //
                zOrder[1 - mirror.y()][1 - mirror.x()].dualVertex(), //
                zOrder[1 - mirror.y()][0 + mirror.x()].dualVertex()};
    } else {
      vertex = {zOrder[0 + mirror.z()][0 + mirror.y()][0 + mirror.x()].dualVertex(),
                zOrder[0 + mirror.z()][0 + mirror.y()][1 - mirror.x()].dualVertex(),
                zOrder[0 + mirror.z()][1 - mirror.y()][1 - mirror.x()].dualVertex(),
                zOrder[0 + mirror.z()][1 - mirror.y()][0 + mirror.x()].dualVertex(),
                zOrder[1 - mirror.z()][0 + mirror.y()][0 + mirror.x()].dualVertex(),
                zOrder[1 - mirror.z()][0 + mirror.y()][1 - mirror.x()].dualVertex(),
                zOrder[1 - mirror.z()][1 - mirror.y()][1 - mirror.x()].dualVertex(),
                zOrder[1 - mirror.z()][1 - mirror.y()][0 + mirror.x()].dualVertex()};
    }

    int index = 0;
    ForEach<powN<dim>(2)>([&](auto i) {
      if (vertex[i].value() > isoValue)
        index += (1 << i);
    });
    if (index == 0 || index == powN<dim - 1>(16) - 1)
      return;

    for (const int8_t *edge = MarchingCubesCases<dim>()[index]; *edge > -1; edge += dim) {
      Vec<Real, dim> triVertices[dim];
      for (int ii = 0; ii < dim; ii++) {
        const int8_t *vert = MarchingCubes_edges<dim>()[edge[ii]];
        const DualVertex<dim> v0 = vertex[vert[0]];
        const DualVertex<dim> v1 = vertex[vert[1]];
        const float t = (isoValue - v0.value()) / (v1.value() - v0.value());
        triVertices[ii] = Lerp(v0.pos(), v1.pos(), t);
      }

      bool success = true;
      ForEach<dim>([&](auto i) {
        constexpr int j = (i + 1) % dim;
        if (triVertices[i] == triVertices[j])
          success = false;
      });
      if (!success)
        continue;

      f(triVertices);
    }
  }
};

} // namespace ARIA
