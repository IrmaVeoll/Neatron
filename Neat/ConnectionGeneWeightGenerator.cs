using Neat.Utils;

namespace Neat
{
    public static class ConnectionGeneWeightGenerator
    {
        public static float GetRandomUniform() =>
            RandomSource.Range(-ConnectionGene.MaxAbsWeightValue, ConnectionGene.MaxAbsWeightValue);
    }
}