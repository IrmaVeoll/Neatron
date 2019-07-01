namespace Neat
{
    public interface INeatChromosomesDistance
    {
        void Prepare();
        void HandleMatchedGenes(in ConnectionGene cg1, in ConnectionGene cg2);
        void HandleExcessGene(in ConnectionGene cg);
        void HandleDisjointGene(in ConnectionGene cg);
        float GetDistance();
    }
}