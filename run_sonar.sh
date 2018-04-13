#!/usr/bin/env bash

sonar-scanner \
    -Dsonar.projectKey=YOPO-Darkflow \
    -Dsonar.sources=. \
    -Dsonar.host.url=https://sonarcloud.io \
    -Dsonar.organization=rij12-github \
    -Dsonar.login=b7bd23cbad5d2a534003dbdbdecf517265a5e739