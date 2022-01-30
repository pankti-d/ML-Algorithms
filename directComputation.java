public class directComputation {

    static int N = 2;
    static int M = 3;
    static int T = 4;

    // Let states 0=H and 1=C
    static int[][]  states = {{0,0,0,0},{0,0,0,1},{0,0,1,0},{0,0,1,1},{0,0,1,0},{0,1,0,1},{0,1,1,1},{0,1,1,0},{1,1,1,1},{1,1,1,0},{1,1,0,1},{1,1,0,0},
            {1,0,1,1},{1,0,1,0},{1,0,0,0},{1,0,0,1}};

    static double[][] A = {{0.7,0.3},{0.4,0.6}};
    static double[][] B = {{0.1,0.4,0.5},{0.7,0.2,0.1}};
    static double[] pi = {0.6,0.4};

    static int[] observations = new int[T];

    static double total = 0;

    static void directSum(){

        double probability = 0;

        for(int i = 0 ; i<16 ; i++)
        {
            probability = pi[states[i][0]]*B[states[i][0]][observations[0]]*A[states[i][0]][states[i][1]]*
                    B[states[i][1]][observations[1]]*A[states[i][1]][states[i][2]]*B[states[i][2]][observations[2]]*
                    A[states[i][2]][states[i][3]]*B[states[i][3]][observations[3]];
            total += probability;
        }
    }

    public static void main(String[] args) {

        for (int o0 = 0; o0 <M; o0++) {

            for (int o1 = 0; o1 < M; o1++) {

                for (int o2 = 0; o2 < M ; o2++) {

                    for (int o3 = 0; o3 < M; o3++) {

                        observations[0] = o0;
                        observations[1] = o1;
                        observations[2] = o2;
                        observations[3] = o3;
                        directSum();
                        System.out.println(observations[0]+","+observations[1]+","+observations[2]+","+observations[3]+"\t\t"+total);
                    }

                }

            }

        }
        System.out.println("Sum of probabilities by direct computation method\t"+ total);
    }
}
