{
  description = "Delivery";
  inputs = {
    nix-ros-overlay.url = "github:lopsided98/nix-ros-overlay/develop";
    nixpkgs.follows = "nix-ros-overlay/nixpkgs";
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
  };
  outputs = { self, nix-ros-overlay, nixpkgs, nixpkgs-unstable }:
    nix-ros-overlay.inputs.flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ 
            nix-ros-overlay.overlays.default
          ];
        };

        unstablePkgs = import nixpkgs-unstable {
          inherit system;
          config.allowUnfree = true;
        };

        # Pull the specific Python interpreter ROS is using
        pythonWithPackages = pkgs.rosPackages.${rosDistro}.python3.withPackages (p: with p; [
          numpy
          scipy
          opencv4
          debugpy
          osmnx
          shapely
          jax
          optax
        ]);

        rosDistro = "lyrical";

        # gz-gui 10's QML (GzSnackBar.qml) imports Qt5Compat.GraphicalEffects.
        qt5compat = pkgs.qt6Packages.qt5compat;
      in {
        devShells.default = pkgs.mkShell {
          name = "Delivery";
          packages = with pkgs; [
            colcon
            opencv
            pythonWithPackages
            qt5compat
            unstablePkgs.foxglove-studio

            (with rosPackages.${rosDistro}; buildEnv {
              paths = [
                ament-cmake
                ament-cmake-core
                ament-cmake-python
                python-cmake-module
                ros-core
                rclcpp
                rclpy
                rviz2
                cv-bridge
                joy
                joy-linux
                joy-teleop
                ros-gz-bridge
                ros-gz-interfaces
                ros-gz-sim
                gz-sim-vendor
                gz-tools-vendor
                xacro
                teleop-twist-keyboard
                robot-state-publisher
                foxglove-bridge
                tf2-tools
              ];
            })
          ];

          # qt5compat installs to its own store path; Qt only searches the
          # qtbase prefix, so point the QML engine at it explicitly.
          QML2_IMPORT_PATH = "${qt5compat}/lib/qt-6/qml";
        };
      });
  nixConfig = {
    extra-substituters = [ "https://ros.cachix.org" ];
    extra-trusted-public-keys = [ "ros.cachix.org-1:dSyZxI8geDCJrwgvCOHDoAfOm5sV1wCPjBkKL+38Rvo=" ];
  };
}
