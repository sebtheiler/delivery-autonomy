{
  description = "Delivery";
  inputs = {
    nix-ros-overlay.url = "github:lopsided98/nix-ros-overlay/develop";
    # need to use develop since ros-gz-bridge is not yet available on master
    nixpkgs.follows = "nix-ros-overlay/nixpkgs";
  };
  outputs = { self, nix-ros-overlay, nixpkgs }:
    nix-ros-overlay.inputs.flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ 
            nix-ros-overlay.overlays.default
          ];
        };
        rosDistro = "humble";

        pythonWithPackages = pkgs.python312.withPackages (p: with p; [
          numpy
          scipy
          opencv4
          debugpy
        ]);

      in {
        devShells.default = pkgs.mkShell {
          name = "WUAIR";
          packages = with pkgs; [
            colcon
            opencv
            pythonWithPackages

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
                xacro
                teleop-twist-keyboard
                robot-state-publisher
              ];
            })
          ];
        };
      });
  nixConfig = {
    extra-substituters = [ "https://ros.cachix.org" ];
    extra-trusted-public-keys = [ "ros.cachix.org-1:dSyZxI8geDCJrwgvCOHDoAfOm5sV1wCPjBkKL+38Rvo=" ];
    permittedInsecurePackages = [
      "freeimage-3.18.0-unstable-2024-04-18"
    ];
  };
}
