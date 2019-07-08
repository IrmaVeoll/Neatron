using System;
using Redzen.Random;

namespace Neat.Utils
{
    public static class RandomSource
    {
        [ThreadStatic]
        private static IRandomSource _rng;
        
        public static IRandomSource Rng => _rng ?? (_rng = new Xoshiro256StarStarRandom());

        public static float Next() => Rng.NextFloat();
        
        public static int Next(int maxValue) => Rng.Next(maxValue);

        public static float Range(float minimum, float maximum) => Rng.NextFloat() * (maximum - minimum) + minimum;

        public static bool NextBool() => Rng.NextBool();
    }
}