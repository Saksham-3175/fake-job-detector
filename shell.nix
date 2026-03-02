with import <nixpkgs> {};

mkShell {
  buildInputs = [
    python312
    python312Packages.pip
    python312Packages.numpy
    python312Packages.pandas    
    python312Packages.scikit-learn
    python312Packages.mlflow
    python312Packages.joblib
    python312Packages.matplotlib
    python312Packages.seaborn
    python312Packages.jobspy
    zlib
    stdenv.cc.cc.lib
  ];
}