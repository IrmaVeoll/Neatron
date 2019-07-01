using System;
using static Neat.NeuronGeneType;

namespace Neat
{
    public sealed class NeuronGene : IEquatable<NeuronGene>
    {
        internal NeuronGene(NeuronGeneType neuronGeneType, int id)
        {
            NeuronGeneType = neuronGeneType;
            Id = id;
        }
        
        private NeuronGeneType NeuronGeneType { get; }

        public bool IsHidden => NeuronGeneType == Hidden;
        
        public bool IsOutput => NeuronGeneType == Effector;

        public bool IsInput => IsBias || NeuronGeneType == Sensor;
        
        public bool IsBias => NeuronGeneType == Bias;
        
        public int Id { get; }

        public override string ToString()
        {
            string NeuronGeneTypeToString()
            {
                switch (NeuronGeneType)
                {
                    case Sensor:
                        return "s";
                    case Bias:
                        return "b";
                    case Hidden:
                        return "h";
                    case Effector:
                        return "e";
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
            return $"{Id}{NeuronGeneTypeToString()}";
        }

        public bool Equals(NeuronGene other)
        {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            return Id == other.Id;
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != GetType()) return false;
            return Equals((NeuronGene) obj);
        }
        
        public static bool operator == (NeuronGene leftSide, NeuronGene rightSide)
        {
            return leftSide?.Equals(rightSide) ?? ReferenceEquals(rightSide, null);
        }

        public static bool operator != (NeuronGene leftSide, NeuronGene rightSide) 
        {
            return !(leftSide == rightSide);
        }

        public override int GetHashCode() => Id.GetHashCode();
    }
}