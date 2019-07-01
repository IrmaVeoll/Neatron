namespace Neat
{
    public readonly struct NetworkConnection
    {
        public readonly float Weight;
        public readonly int TargetIdx;

        public NetworkConnection(int targetIdx, float weight)
        {
            Weight = weight;
            TargetIdx = targetIdx;
        }

        public override string ToString()
        {
            return $"TargetIdx: {TargetIdx}, Weight: {Weight:+0.0000;-0.0000}";
        }
    }
}