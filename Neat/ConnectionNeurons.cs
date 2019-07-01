namespace Neat
{
    public sealed class ConnectionNeurons
    {
        internal ConnectionNeurons(NeuronGene sourceNeuronGene, NeuronGene targetNeuronGene)
        {
            Source = sourceNeuronGene;
            Target = targetNeuronGene;
        }

        internal readonly NeuronGene Source;
        internal readonly NeuronGene Target;

        public int SourceId => Source.Id;
        public int TargetId => Target.Id;

        public bool IsOuterLayersConnection => !(Source.IsHidden || Target.IsHidden);
        
        public override bool Equals(object obj) => obj is ConnectionNeurons connectedNeurons && Equals(connectedNeurons);

        private bool Equals(ConnectionNeurons c) => SourceId == c.SourceId && TargetId == c.TargetId;

        public override int GetHashCode()
        {
            var hash = 17;
            hash = hash * 23 + Source.GetHashCode();
            hash = hash * 23 + Target.GetHashCode();
            return hash;
        }
        
        public override string ToString() => $"{Source} -> {Target}";
    }
}