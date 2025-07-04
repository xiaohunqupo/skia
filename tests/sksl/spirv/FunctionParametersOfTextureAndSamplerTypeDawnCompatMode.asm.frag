               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %sk_FragColor %c
               OpExecutionMode %main OriginUpperLeft

               ; Debug Information
               OpName %sk_FragColor "sk_FragColor"  ; id %5
               OpName %aTexture "aTexture"          ; id %9
               OpName %aSampledTexture_texture "aSampledTexture_texture"    ; id %12
               OpName %aSampledTexture_sampler "aSampledTexture_sampler"    ; id %15
               OpName %c "c"                                                ; id %18
               OpName %helpers_helper_h4ZT_aSampledTexture "helpers_helper_h4ZT_aSampledTexture"    ; id %2
               OpName %helper_h4TZ_aSampledTexture "helper_h4TZ_aSampledTexture"                    ; id %3
               OpName %main "main"                                                                  ; id %4

               ; Annotations
               OpDecorate %helpers_helper_h4ZT_aSampledTexture RelaxedPrecision
               OpDecorate %helper_h4TZ_aSampledTexture RelaxedPrecision
               OpDecorate %sk_FragColor RelaxedPrecision
               OpDecorate %sk_FragColor Location 0
               OpDecorate %sk_FragColor Index 0
               OpDecorate %aTexture Binding 1
               OpDecorate %aTexture DescriptorSet 0
               OpDecorate %aSampledTexture_texture Binding 2
               OpDecorate %aSampledTexture_texture DescriptorSet 0
               OpDecorate %aSampledTexture_sampler Binding 3
               OpDecorate %aSampledTexture_sampler DescriptorSet 0
               OpDecorate %c Location 1
               OpDecorate %22 RelaxedPrecision
               OpDecorate %25 RelaxedPrecision
               OpDecorate %33 RelaxedPrecision

               ; Types, variables and constants
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%sk_FragColor = OpVariable %_ptr_Output_v4float Output  ; RelaxedPrecision, Location 0, Index 0
         %10 = OpTypeImage %float 2D 0 0 0 2 R32f
%_ptr_UniformConstant_10 = OpTypePointer UniformConstant %10
   %aTexture = OpVariable %_ptr_UniformConstant_10 UniformConstant  ; Binding 1, DescriptorSet 0
         %13 = OpTypeImage %float 2D 0 0 0 1 Unknown
%_ptr_UniformConstant_13 = OpTypePointer UniformConstant %13
%aSampledTexture_texture = OpVariable %_ptr_UniformConstant_13 UniformConstant  ; Binding 2, DescriptorSet 0
         %16 = OpTypeSampler
%_ptr_UniformConstant_16 = OpTypePointer UniformConstant %16
%aSampledTexture_sampler = OpVariable %_ptr_UniformConstant_16 UniformConstant  ; Binding 3, DescriptorSet 0
    %v2float = OpTypeVector %float 2
%_ptr_Input_v2float = OpTypePointer Input %v2float
          %c = OpVariable %_ptr_Input_v2float Input     ; Location 1
         %21 = OpTypeFunction %v4float %_ptr_UniformConstant_13 %_ptr_UniformConstant_10
         %29 = OpTypeSampledImage %13
         %31 = OpTypeFunction %v4float %_ptr_UniformConstant_10 %_ptr_UniformConstant_13
       %void = OpTypeVoid
         %37 = OpTypeFunction %void


               ; Function helpers_helper_h4ZT_aSampledTexture
%helpers_helper_h4ZT_aSampledTexture = OpFunction %v4float None %21     ; RelaxedPrecision
         %22 = OpFunctionParameter %_ptr_UniformConstant_13             ; RelaxedPrecision
         %23 = OpFunctionParameter %_ptr_UniformConstant_10

         %24 = OpLabel
         %26 =   OpLoad %13 %22
         %27 =   OpLoad %16 %aSampledTexture_sampler
         %28 =   OpSampledImage %29 %26 %27
         %30 =   OpLoad %v2float %c
         %25 =   OpImageSampleImplicitLod %v4float %28 %30  ; RelaxedPrecision
                 OpReturnValue %25
               OpFunctionEnd


               ; Function helper_h4TZ_aSampledTexture
%helper_h4TZ_aSampledTexture = OpFunction %v4float None %31     ; RelaxedPrecision
         %32 = OpFunctionParameter %_ptr_UniformConstant_10
         %33 = OpFunctionParameter %_ptr_UniformConstant_13     ; RelaxedPrecision

         %34 = OpLabel
         %35 =   OpFunctionCall %v4float %helpers_helper_h4ZT_aSampledTexture %33 %32
                 OpReturnValue %35
               OpFunctionEnd


               ; Function main
       %main = OpFunction %void None %37

         %38 = OpLabel
         %39 =   OpFunctionCall %v4float %helper_h4TZ_aSampledTexture %aTexture %aSampledTexture_texture
                 OpStore %sk_FragColor %39
                 OpReturn
               OpFunctionEnd
