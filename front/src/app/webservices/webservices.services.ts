import { Component, OnInit, Injectable } from '@angular/core';
import { AuthenticationService } from '../authentication';
import { Router } from '@angular/router';
import { Http, Response } from '@angular/http';

@Injectable()
export class WebService {
  constructor(private authService: AuthenticationService) { }

  public getDataFromBackend(body:object) {
   
    //return this.authService.getResource('/api/protected');
    return this.authService.getResource('/api/chat',body);
  }

  public getImageDataFromBackend(body:object) {
   
    //return this.authService.getResource('/api/protected');
    return this.authService.getResource('/api/image',body);
  }

  public isAuthenticated() {
  //  if (!this.authService.isAuthenticated()) {
    //  this.authService.clearUserDataAndRedirect();
    //}
  }
}
