import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Arrays;


public class englishHMM {

    //For reading file from the folder character by character
    static List<Integer> O = new ArrayList<>();

    public static int generateObservationArray() {
        String filePath = "E:\\SJSU - MSCS\\CS271 - Topics in ML\\Assignments - Source Code\\brown.txt";

        File f = new File(filePath);
        FileReader fileReader = null;
        try {
            fileReader = new FileReader(f);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        BufferedReader bufferedReader = new BufferedReader(fileReader);
        int c = 0;
        try {
            while ((c = bufferedReader.read()) != -1 && O.size() < 50000) {
                char character = (char) c;
                if (Character.isAlphabetic(character)) {
                    character = Character.toLowerCase(character);
                    O.add(character - 97);
                } else if (character == ' ') {
                    O.add(26);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return O.size();
    }

    static int N = 26;
    static int M = 27;
    static int T = generateObservationArray();
    static double[][] A = new double[N][N];
    static double[][] B = new double[N][M];
    static double[] pi = new double[N];
    static double[][] alpha = new double[T][N];
    static double[][] beta = new double[T][N];
    static double[][] gamma = new double[T][N];
    static double[][][] diGamma = new double[T][N][N];
    static int maxIters = 200;
    static int iters = 0;
    static double logProb = 0;
    static double oldLogProb = Double.NEGATIVE_INFINITY;
    static double c[] = new double[T];

    public static void initialization() {

        Random random = new Random();

        // A matrix initialization
        for (int i = 0; i < N; i++) {
            int rowSum = 0;
            for (int j = 0; j < N; j++) {
                int randomNumber = random.nextInt(100);
                rowSum += randomNumber;
                A[i][j] = randomNumber;
            }
            // Take sum of all elements in a row and divide each by sum
            for (int j = 0; j < N; j++) {
                A[i][j] = A[i][j] / rowSum;
            }
        }
        // B matrix initialization
        for (int i = 0; i < N; i++) {
            int rowSum = 0;
            for (int j = 0; j < M; j++) {
                int randomNumber = random.nextInt(100);
                rowSum += randomNumber;
                B[i][j] = randomNumber;
            }
            for (int j = 0; j < M; j++) {
                B[i][j] = B[i][j] / rowSum;
            }
        }

        // Pi matrix initialization
        int sum = 0;
        for (int i = 0; i < N; i++) {
            int randomNumber = random.nextInt(100);
            sum += randomNumber;
            pi[i] = randomNumber;
        }
        for (int i = 0; i < N; i++) {
            pi[i] = pi[i] / sum;
        }
    }

    public static void PrintA_B_pi() {
        System.out.println("A");
        System.out.println(Arrays.deepToString(A));
        System.out.println("B");
        System.out.println(Arrays.deepToString(B));
        System.out.println("pi" + pi.length);
        System.out.println(Arrays.toString(pi));
    }

    // Alpha Pass/ Forward Algorithm
    public static void alphaPassCompute() {
        c[0] = 0;
        //Compute alpha(0)(i)
        for (int i = 0; i < N; i++) {

            alpha[0][i] = pi[i] * B[i][O.get(0)];
            c[0] = c[0] + alpha[0][i];
        }

        c[0] = 1 / c[0];

        for (int i = 0; i < N; i++) {

            alpha[0][i] = alpha[0][i] * c[0];
        }
        // Compute alpha(t)(i)
        for (int t = 1; t < T; t++) {
            c[t] = 0;

            for (int i = 0; i < N; i++) {
                alpha[t][i] = 0;

                for (int j = 0; j < N; j++) {
                    alpha[t][i] = alpha[t][i] + alpha[t - 1][j] * A[j][i];
                }
                alpha[t][i] = alpha[t][i] * B[i][O.get(t)];
                c[t] = c[t] + alpha[t][i];
            }
            c[t] = 1 / c[t];

            for (int i = 0; i < N; i++) {
                alpha[t][i] = c[t] * alpha[t][i];
            }
        }

    }

    public static void betaPassCompute() {
        //backward (beta pass)
        for (int i = 0; i < N; i++) {
            beta[T - 1][i] = c[T - 1];
        }

        for (int t = T - 2; t >= 0; t--) {
            for (int i = 0; i < N; i++) {
                beta[t][i] = 0;

                for (int j = 0; j < N; j++) {
                    beta[t][i] = beta[t][i] + A[i][j] * B[j][O.get(t + 1)] * beta[t + 1][j];
                }

                beta[t][i] = c[t] * beta[t][i];
            }
        }
    }

    public static void gammaDigammaCompute() {
        for (int t = 0; t < T - 1; t++) {

            for (int i = 0; i < N; i++) {

                gamma[t][i] = 0;
                for (int j = 0; j < N; j++) {

                    diGamma[t][i][j] = alpha[t][i] * A[i][j] * B[j][O.get(t + 1)] * beta[t + 1][j];
                    gamma[t][i] = gamma[t][i] + diGamma[t][i][j];
                }
            }
        }
        for (int i = 0; i < N; i++) {
            gamma[T - 1][i] = alpha[T - 1][i];
        }
    }

    public static void reEstimateA_B_Pi() {
        //Restimate pi
        for (int i = 0; i < N; i++) {
            pi[i] = gamma[0][i];
        }

        //Resimate A

        for (int i = 0; i < N; i++) {
            double denom = 0;
            for (int t = 0; t < T - 1; t++) {
                denom = denom + gamma[t][i];
            }
            for (int j = 0; j < N; j++) {
                double numer = 0;

                for (int t = 0; t < T - 1; t++) {
                    numer = numer + diGamma[t][i][j];
                }
                A[i][j] = numer / denom;
            }
        }

        //Resimate B

        for (int i = 0; i < N; i++) {
            double denom = 0;
            for (int t = 0; t < T; t++) {
                denom = denom + gamma[t][i];
            }
            for (int j = 0; j < M; j++) {
                double numer = 0;
                for (int t = 0; t < T; t++) {
                    if (O.get(t) == j) {
                        numer = numer + gamma[t][i];
                    }
                }
                B[i][j] = numer / denom;
            }
        }
    }

    public static void computeLog() {
        // after restimating A, B and pi computing new log
        logProb = 0;
        for (int i = 0; i < T; i++) {
            logProb = logProb + Math.log(c[i]);
        }
        logProb = -1 * logProb;
    }

    public static void iterations() {
        iters++;
        if (iters < maxIters && logProb > oldLogProb) {
            oldLogProb = logProb;
            reCompute();
        } else {
            //printing final A, B and Pi
            System.out.println("Final log\t" + logProb);
            System.out.println("After\t" + iters + "\titerations");
            System.out.println("Final A" + Arrays.deepToString(A));
            System.out.println("Final B" + Arrays.deepToString(B));
            System.out.println("Final pi" + Arrays.toString(pi));
        }
    }

    public static void reCompute() {
        alphaPassCompute();
        betaPassCompute();
        gammaDigammaCompute();
        reEstimateA_B_Pi();
        computeLog();
        iterations();
    }

    public static void main(String[] args) {
        //main
        initialization();
        System.out.println("Initial Values of A, B and pi");
        PrintA_B_pi();
        reCompute();
    }

}

