{
  description = "Dev Environment for AL\ML\DS\NLP";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {allowUnfree = true;};
    };
  in {
    devShells.${system}.default = pkgs.mkShell {
      name = "development-of-intelligent";

      packages =
        (with pkgs; [
          stdenv.cc.cc.lib
          git-crypt
          stdenv.cc.cc # jupyter lab needs
          # sometimes you might need something additional like the following - you will get some useful error if it is looking for a binary in the environment.
          taglib
          openssl
          git
          libxml2
          libxslt
          libzip
          zlib
          poetry
          python311
        ])
        ++ (with pkgs.python311Packages; [
          virtualenv
          pip
          ipykernel
          jupyterlab
          pyzmq
          venvShellHook
          pip
          numpy
          pandas
          pandas-datareader
          matplotlib
          requests
          seaborn
          openpyxl
          cufflinks
          plotly
          scikit-learn
          imbalanced-learn
          tensorflow
          keras
        ]);

      env = {
        LD_LIBRARY_PATH =
          pkgs.lib.makeLibraryPath [
          ];

        JUPYTER_CONFIG_DIR = "./.jupyter";
        TF_ENABLE_ONEDNN_OPTS = 0;
      };

      shellHook = ''
        echo "Entering devShell for ${system}";

        # Persistent virtual environment setup
        if [[ ! -d ./venv ]]; then
          python -m venv ./venv
        fi
        source ./venv/bin/activate

        # workaround for vscode's to find the venv
        venv="$(cd $(dirname $(which python)); cd ..; pwd)"
        ln -Tsf "$venv" .venv
      '';
    };
  };
}
