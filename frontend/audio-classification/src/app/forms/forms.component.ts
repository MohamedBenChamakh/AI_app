import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup } from '@angular/forms';
import { catchError, tap } from 'rxjs';


@Component({
  selector: 'app-forms',
  templateUrl: './forms.component.html',
  styleUrls: ['./forms.component.scss'],
})
export class FormsComponent implements OnInit {
  form!: FormGroup;
  result:string="";

  
  constructor(
    private httpClient: HttpClient,
    private formBuilder: FormBuilder
  ) {}

  ngOnInit(): void {
    this.initForm();
  }

  onFileSelect(event:any) {
    if (event.target.files.length > 0) {
      const file = event.target.files[0];
      this.form.get('file').setValue(file);
    }
  }

  initForm(): void {
    this.form = this.formBuilder.group({
      algo:'',
      file:['']
    });
  }
  onSubmit(): void {

    const formData = new FormData();
    formData.append('file', this.form.get('file').value);
    console.log(this.form.value);
    
    this.httpClient.post("http://127.0.0.1:4200/api/"+this.form.value['algo'],formData).subscribe(
      (response: any) => {
        console.log(response);
        this.result="Prédiction : "+response.genre;

      },
    )
  }
}
