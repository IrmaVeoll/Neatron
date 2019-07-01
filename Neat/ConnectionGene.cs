namespace Neat
{
    public readonly struct ConnectionGene
    {
        public const float MaxAbsWeightValue = 5;
        
        internal ConnectionGene(ConnectionNeurons connectionNeurons, float weight, int id)
        {
            ConnectionNeurons = connectionNeurons;
            Weight = weight;
            Id = id;
        }

        public int Id { get; }

        public ConnectionNeurons ConnectionNeurons { get; }

        public float Weight { get; }

        public bool IsOuterLayersConnection => ConnectionNeurons.IsOuterLayersConnection;

        public NeuronGene Source => ConnectionNeurons.Source;
        public NeuronGene Target => ConnectionNeurons.Target;

        public int SourceId => ConnectionNeurons.SourceId;
        public int TargetId => ConnectionNeurons.TargetId;

        public override string ToString() => $"{ConnectionNeurons} Weight={Weight:+0.0000;-0.0000} ({Id})";
    }
}