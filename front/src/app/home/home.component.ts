import { Component, OnInit, OnDestroy } from '@angular/core';
import { Http, Response } from '@angular/http';
import { AuthenticationService } from '../authentication';
import { Router } from '@angular/router';
import { NavbarComponent } from '../navbar';
import { WebService } from '../webservices';

@Component({
  selector: 'home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css'],
  providers: [WebService, AuthenticationService]
})
export class HomeComponent implements OnInit, OnDestroy {

  public heroes = '';
  myText:string;
  myImage:string;
  myImageDesc:any;
  imageLocalPath:string = '/Users/ajinkya.parkar@ibm.com/Documents/deep/flaskAngular2/front/src/';

  constructor(private http: Http, private router: Router, private webservice: WebService) { }

  public ngOnInit() {
    //this.webservice.isAuthenticated();
  }

  public ngOnDestroy() {
    // Will clear when component is destroyed e.g. route is navigated away from.
    console.log('destroyed');
  }

  public clear() {
    this.heroes = '';
  }

  /**
   * Fetch the data from the python-flask backend
   */
  public getData() {
    document.getElementById("myChat").innerHTML += '<li style="width:100%;">' +
    '<div class="msj-rta macro">' +
        '<div class="text text-r">' +
            '<p>'+this.myText+'</p>' +
            '<p><small></small></p>' +
        '</div>' +
    '<div class="avatar" style="padding:0px 0px 0px 10px !important"><img class="img-circle" style="width:100%;" src="" /></div>' +                                
'</li>';
    let body = {
      myText: this.myText
    };
    this.myText = "";

    this.webservice.getDataFromBackend(body)
      .subscribe(
        
      (data) => {
        this.heroes = data;
        document.getElementById("myChat").innerHTML += '<li style="width:100%">' +
        '<div class="msj macro">' +
        '<div class="avatar"><img class="img-circle" style="width:100%;" src="" /></div>' +
            '<div class="text text-l">' +
                '<p>'+this.heroes+'</p>' +
                '<p><small></small></p>' +
            '</div>' +
        '</div>' +
    '</li>';
        //this.handleData(data);
      },
      (err) => {
     
        this.logError(err)},
      () => console.log('got data')
      );
  }

  public getImage() {
    
    let body = {
      myText: this.imageLocalPath +this.myImage
    };
    
    debugger;
    this.webservice.getImageDataFromBackend(body)
      .subscribe(
        
      (data) => {
        debugger;
        this.myImageDesc = data;
        //document.getElementById("myChat").innerHTML += '<li style="width:100%">i</li>';
        //this.handleData(data);
      },
      (err) => {
     
        this.logError(err)},
      () => console.log('got data')
      );
  }

  private handleData(data: Response) {
    
   // if (data.status === 200) {
      let receivedData = data;
      this.heroes = data.json();
    //}
    console.log(data.json());
  }

  private logError(err: Response) {
    console.log('There was an error: ' + err.status);
    if (err.status === 0) {
      console.error('Seems server is down');
    }
    if (err.status === 401) {
      this.router.navigate(['/sessionexpired']);
    }
  }
}
