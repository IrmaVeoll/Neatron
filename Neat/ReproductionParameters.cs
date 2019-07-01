using System.Collections.Generic;
using static Neat.CrossoverType;

namespace Neat
{
    public enum CrossoverType
    {
        OnePoint,
        TwoPoints,
        Uniform,
        ArithmeticRecombination
    }
    
    public abstract class WeightMutationInfo
    {
        public float RouletteWheelShare { get; set; }
        
        public int ConnectionCount { get; set; } = 1;
    }

    public sealed class WeightPerturb : WeightMutationInfo {}

    public sealed class WeightTweak : WeightMutationInfo
    {
        public float Sigma { get; set; } = 0.05f;
    }

    public sealed class WeightMutations
    {
        public float OverallRouletteWheelShare { get; set; }
        
        public List<WeightMutationInfo> Mutations = new List<WeightMutationInfo>{
            new WeightTweak
            {
                RouletteWheelShare = 1f,
            }
        };
    }

    public sealed class ReproductionParameters
    {
        public CrossoverType CrossoverType { get; set; } = OnePoint;

        public WeightMutations WeightMutations { get; set; } =
            new WeightMutations { OverallRouletteWheelShare = 0.95f };
        public float SplitConnectionRouletteWheelShare { get; set; } = 0.01f;
        public float AddConnectionRouletteWheelShare { get; set; } = 0.02f;
        public float RemoveConnectionRouletteWheelShare { get; set; } = 0.02f;
    }
}