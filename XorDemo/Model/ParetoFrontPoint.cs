using System;
using Neat;

namespace XorDemo.Model
{
    public class ParetoFrontPoint : IComparable<ParetoFrontPoint>
    {
        internal ParetoFrontPoint(Genome genome, float fitness, int complexity)
        {
            Genome = genome;
            Fitness = fitness;
            SharedFitness = Fitness;
            Complexity = complexity;
            Simplicity = complexity == 0 ? 1f : (1f / complexity);
        }

        public Genome Genome { get; }
        public float Fitness { get; }
        public float SharedFitness { get; set; }
        public float Simplicity { get; }
        public int Complexity { get; }
        public int Rank { get; set; } = -1;
        public float Sparsity { get; set; }
        public float CentroidDistance { get; set; }

        public bool IsBetterThan(ParetoFrontPoint other)
        {
            if (this == other) return false;

            if (Rank < other.Rank) return true;
            if (Rank > other.Rank) return false;

            if (Sparsity < other.Sparsity) return false;
            if (Sparsity > other.Sparsity) return true;

            //if (SharedFitness > other.SharedFitness) return true;
            //if (SharedFitness < other.SharedFitness) return false;
            
            if (Simplicity > other.Simplicity) return true;
            if (Simplicity < other.Simplicity) return false;
            
//            Genome.Network.Reset();
//            
//            var sensors = Genome.Network.Sensors;
//            
//            sensors[0] = 1;
//            sensors[1] = 1;
//
//            Genome.Network.Activate();
//
//            var thisRes = Genome.Network.Effectors[0];
//            
//            other.Genome.Network.Reset();
//            
//            var otherSensors = other.Genome.Network.Sensors;
//            
//            otherSensors[0] = 1;
//            otherSensors[1] = 1;
//
//            other.Genome.Network.Activate();
//
//            var otherRes = other.Genome.Network.Effectors[0];
//
//            var thisFitness = XorNetworkSearch.Evaluate(Genome);
//            var otherFitness = XorNetworkSearch.Evaluate(other.Genome);
//            
//            Debug.Assert(thisRes.Equals(otherRes));
//            Debug.Assert(thisFitness.Equals(otherFitness));

            return false;
        }

        public bool IsDominatedBy(ParetoFrontPoint other)
        {
            return other.Fitness >= Fitness &&
                   other.Simplicity > Simplicity ||
                   other.Fitness > Fitness &&
                   other.Simplicity >= Simplicity;
        }

        public int CompareTo(ParetoFrontPoint other) => CentroidDistance.CompareTo(other.CentroidDistance);

        public override string ToString()
        {
            return $"Fitness: {Fitness:0.0000} Simplicity: {Simplicity:0.00} Rank: {Rank} Sparsity: {Sparsity:0.0000} Complexity: {Complexity}";
        }
    }
}